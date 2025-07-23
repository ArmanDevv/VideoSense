import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper 
from transformers import AutoTokenizer
import argparse
import requests
import yt_dlp
import tempfile
import json
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


class AudioProcessor:
    def extract_features(self, video_path, max_length=300):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir="/tmp"):
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(
            temp_dir, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed: " + segment_path)

        return segment_path


class YouTubeDownloader:
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def download_video(self, video_url, max_duration=600):  # 10 minutes max
        try:
            # Create unique filename
            temp_video_path = os.path.join(self.temp_dir, f"temp_video_{os.getpid()}.mp4")
            
            # yt-dlp options with anti-bot protection
            ydl_opts = {
                'format': 'worst[height<=480][ext=mp4]/worst[ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]',  # Prefer lower quality first
                'outtmpl': temp_video_path,
                'noplaylist': True,
                'extract_flat': False,
                'writethumbnail': False,
                'writeinfojson': False,
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
                'match_filter': self._duration_filter(max_duration),
                
                # Anti-bot protection measures
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'referer': 'https://www.youtube.com/',
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Keep-Alive': '300',
                    'Connection': 'keep-alive',
                },
                
                # Rate limiting to appear more human-like  
                'sleep_interval': 1,
                'max_sleep_interval': 3,
                'sleep_interval_subtitles': 1,
                
                # Retry settings
                'retries': 3,
                'fragment_retries': 3,
                'retry_sleep_functions': {'http': lambda n: 2 ** n},
                
                # Additional anti-detection measures
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'logtostderr': False,
                'extract_comments': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Get video info first
                    print(f"Extracting video info from: {video_url}", file=sys.stderr)
                    info = ydl.extract_info(video_url, download=False)
                    duration = info.get('duration', 0)
                    
                    if duration > max_duration:
                        raise ValueError(f"Video too long: {duration}s > {max_duration}s")
                    
                    print(f"Starting download: {info.get('title', 'Unknown')} ({duration}s)", file=sys.stderr)
                    
                    # Add a small delay before download
                    import time
                    time.sleep(1)
                    
                    # Download the video
                    ydl.download([video_url])
                    
                except yt_dlp.DownloadError as e:
                    error_msg = str(e).lower()
                    if 'sign in' in error_msg or 'bot' in error_msg:
                        raise ValueError("YouTube has detected automated access. This is common with server-based applications. Please try again later or use a different video.")
                    elif 'private' in error_msg or 'unavailable' in error_msg:
                        raise ValueError("This video is private or unavailable. Please try a different public video.")
                    elif 'blocked' in error_msg or 'restricted' in error_msg:
                        raise ValueError("This video is geographically restricted or blocked. Please try a different video.")
                    else:
                        raise ValueError(f"YouTube download failed: {str(e)}")
            
            if not os.path.exists(temp_video_path):
                raise ValueError("Download failed - video file was not created")
                
            print(f"Video successfully downloaded to: {temp_video_path}", file=sys.stderr)
            return temp_video_path
            
        except Exception as e:
            # Clean up partial downloads
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass
            
            error_msg = str(e)
            print(f"Download error: {error_msg}", file=sys.stderr)
            raise ValueError(f"YouTube download error: {error_msg}")

    
    def _duration_filter(self, max_duration):
        """Filter function to check video duration before download"""
        def filter_func(info_dict):
            duration = info_dict.get('duration')
            if duration and duration > max_duration:
                return f"Video too long: {duration}s"
            return None
        return filter_func




def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_dir, 'model.pth')
    
    if not os.path.exists(model_path):
        print(f"Script directory: {script_dir}", file=sys.stderr)
        print(f"Looking for model at: {model_path}", file=sys.stderr)
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading model from path: {model_path}", file=sys.stderr)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'transcriber': whisper.load_model(
            "base",
            device="cpu" if device.type == "cpu" else device,
        ),
        'device': device
    }


def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']

    result = model_dict['transcriber'].transcribe(
        video_path, word_timestamps=True)

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result["segments"]:
        segment_path = None
        try:
            segment_path = utterance_processor.extract_segment(
                video_path,
                segment["start"],
                segment["end"]
            )

            video_frames = utterance_processor.video_processor.process_video(
                segment_path)
            audio_features = utterance_processor.audio_processor.extract_features(
                segment_path)
            text_inputs = tokenizer(
                segment["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # Move to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # Get predictions
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(
                    outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(
                    sentiment_probs, 3)

            predictions.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "emotions": [
                    {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(emotion_indices, emotion_values)
                ],
                "sentiments": [
                    {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            })

        except Exception as e:
            print(f"Segment failed inference: {str(e)}", file=sys.stderr)

        finally:
            # Cleanup segment file
            if segment_path and os.path.exists(segment_path):
                os.remove(segment_path)
                
    return {"utterances": predictions}


def process_video_from_url(video_url, model_dir="model"):
    """Process video from YouTube URL"""
    downloader = YouTubeDownloader()
    video_path = None
    
    try:
        # Download video
        print(f"Downloading video from: {video_url}", file=sys.stderr)
        video_path = downloader.download_video(video_url)
        print(f"Video downloaded to: {video_path}", file=sys.stderr)
        
        # Load model
        model_dict = model_fn(model_dir)
        
        # Process video
        input_data = {'video_path': video_path}
        predictions = predict_fn(input_data, model_dict)
        
        return predictions
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Cleanup downloaded video
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up: {video_path}", file=sys.stderr)


def process_local_video(video_path, model_dir="model"):
    """Process local video file (existing functionality)"""
    model_dict = model_fn(model_dir)
    input_data = {'video_path': video_path}
    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions["utterances"]:
        print("\nUtterance:", file=sys.stderr)
        print(f"Start: {utterance['start_time']}s, End: {utterance['end_time']}s", file=sys.stderr)
        print(f"Text: {utterance['text']}", file=sys.stderr)
        print("\nTop Emotions:", file=sys.stderr)
        for emotion in utterance['emotions']:
            print(f"{emotion['label']}: {emotion['confidence']:.2f}", file=sys.stderr)
        print("\nTop Sentiments:", file=sys.stderr)
        for sentiment in utterance['sentiments']:
            print(f"{sentiment['label']}: {sentiment['confidence']:.2f}", file=sys.stderr)
        print("-"*50, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Sentiment Analysis')
    parser.add_argument('--video_url', type=str, help='YouTube video URL')
    parser.add_argument('--video_path', type=str, help='Local video file path')
    parser.add_argument('--model_dir', type=str, default='model', help='Model directory')
    
    args = parser.parse_args()
    
    try:
        if args.video_url:
            # Process YouTube URL
            result = process_video_from_url(args.video_url, args.model_dir)
            # Ensure only JSON goes to stdout
            sys.stdout.write(json.dumps(result))
            sys.stdout.flush()
        elif args.video_path:
            # Process local video - this currently doesn't return JSON
            process_local_video(args.video_path, args.model_dir)
            # You might want to modify process_local_video to return JSON too
        else:
            error_result = {"error": "Either --video_url or --video_path must be provided"}
            sys.stdout.write(json.dumps(error_result))
            sys.stdout.flush()
            
    except Exception as e:
        error_result = {"error": str(e)}
        sys.stdout.write(json.dumps(error_result))
        sys.stdout.flush()
        print(f"Exception occurred: {str(e)}", file=sys.stderr)
