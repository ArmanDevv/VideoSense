import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import time
import psutil
import whisper 
from transformers import AutoTokenizer
import argparse
import requests
import yt_dlp
import gc
import tempfile
import json
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        total_mb = psutil.virtual_memory().total / 1024 / 1024
        print(f"Memory at {step_name}: {memory_mb:.1f}MB used, {available_mb:.1f}MB available, {total_mb:.1f}MB total", file=sys.stderr)
    except Exception as e:
        print(f"Memory logging failed: {e}", file=sys.stderr)


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
        
    def download_video(self, video_url, max_duration=600):
        with open("yt_cookies.txt", "w") as f:
            f.write(os.getenv("YTDL_COOKIES", ""))
        try:
            temp_video_path = os.path.join(self.temp_dir, f"temp_video_{os.getpid()}.mp4")
            
            # Enhanced yt-dlp options with cookie simulation
            ydl_opts = {
                'format': 'worst[height<=480]/best[height<=720]/best',  # Start with lowest quality
                'outtmpl': temp_video_path,
                'cookiefile': 'yt_cookies.txt',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
                'match_filter': self._duration_filter(max_duration),
                
                # Advanced anti-detection measures
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'referer': 'https://www.youtube.com/',
                
                # Cookie and session simulation
                'extractor_args': {
                    'youtube': [
                        'player-client=mweb',
                        'po_token=mweb.gvs+mweb.player',
                        'skip=webpage',
                        'player_skip=webpage,configs'
                    ]
                },
                
                # Headers to mimic real browser
                'http_headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Cache-Control': 'max-age=0'
                },
                
                # Rate limiting
                'sleep_interval_requests': 1,
                'sleep_interval': 1,
                'max_sleep_interval': 5,
                
                # Retry configuration  
                'retries': 2,
                'fragment_retries': 2,
                'retry_sleep_functions': {'http': lambda n: min(4 ** n, 30)},
                
                # Additional options
                'nocheckcertificate': True,
                'prefer_insecure': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Add randomized delay before request
                    import time
                    import random
                    time.sleep(random.uniform(2, 5))
                    
                    print(f"Extracting video info: {video_url}", file=sys.stderr)
                    info = ydl.extract_info(video_url, download=False)
                    
                    duration = info.get('duration', 0)
                    title = info.get('title', 'Unknown')
                    
                    if duration > max_duration:
                        raise ValueError(f"Video too long: {duration}s > {max_duration}s")
                    
                    print(f"Starting download yay: {title} ({duration}s)", file=sys.stderr) 
                    
                    # Another small delay before actual download
                    time.sleep(random.uniform(1, 3))
                    
                    ydl.download([video_url])
                    
                except yt_dlp.DownloadError as e:
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['sign in', 'bot', 'verify']):
                        # Try with even more conservative settings
                        return self._download_with_conservative_settings(video_url, max_duration)
                    else:
                        raise ValueError(f"Download failed: {str(e)}")
            
            if not os.path.exists(temp_video_path):
                raise ValueError("Download failed - video file not created")
                
            print(f"Successfully downloaded: {temp_video_path}", file=sys.stderr)
            return temp_video_path
            
        except Exception as e:
            # Cleanup
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass
                    
            raise ValueError(f"YouTube download error: {str(e)}")
        finally:
            # Clean up cookie file
            if os.path.exists('yt_cookies.txt'):
                try:
                    os.remove('yt_cookies.txt')
                except:
                    pass
    
    def _duration_filter(self, max_duration):
        """Filter function to check video duration before download"""
        def filter_func(info_dict):
            duration = info_dict.get('duration')
            if duration and duration > max_duration:
                return f"Video too long: {duration}s"
            return None
        return filter_func

    def _download_with_conservative_settings(self, video_url, max_duration):
        """Ultra-conservative download attempt as fallback"""
        try:
            temp_video_path = os.path.join(self.temp_dir, f"temp_video_conservative_{os.getpid()}.mp4")
            
            # Minimal, conservative options
            ydl_opts = {
                'format': 'worst[ext=mp4]/worst',  # Absolute worst quality
                'outtmpl': temp_video_path,
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
                'retries': 1,
                'fragment_retries': 1,
                'match_filter': self._duration_filter(max_duration),
                
                # Minimal headers
                'user_agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0',
                
                # Slower, more human-like behavior
                'sleep_interval': 3,
                'max_sleep_interval': 10,
            }
            
            print("Attempting conservative download fallback...", file=sys.stderr)
            
            # Wait longer before attempting 
            import time
            time.sleep(10)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
            if os.path.exists(temp_video_path):
                print(f"Conservative download succeeded: {temp_video_path}", file=sys.stderr)
                return temp_video_path
            else:
                raise ValueError("Conservative download also failed - file not created")
                
        except Exception as e:
            # Clean up failed download
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass
            raise ValueError(f"Conservative download failed: {str(e)}")

def download_model_if_missing(model_path, model_url):
    """Download model file with timeout handling"""
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print(f"Model file missing, downloading from Google Drive...", file=sys.stderr)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            session = requests.Session()
            
            # CRITICAL: Add timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Download attempt {attempt + 1}/3...", file=sys.stderr)
                    response = session.get(model_url, stream=True, timeout=60)  # 60 second timeout
                    response.raise_for_status()
                    break
                except (requests.Timeout, requests.ConnectionError) as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}", file=sys.stderr)
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to download after {max_retries} attempts")
                    time.sleep(10)  # Wait before retry
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            print(f"Starting download... Total size: {total_size / (1024*1024):.1f}MB", file=sys.stderr)
            
            # Add download progress timeout
            start_time = time.time()
            max_download_time = 600  # 10 minutes max for large file
            last_progress_time = start_time
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        current_time = time.time()
                        
                        # Check for overall timeout
                        if current_time - start_time > max_download_time:
                            raise TimeoutError("Download timeout exceeded")
                        
                        # Check for stalled download (no progress for 60 seconds)
                        if current_time - last_progress_time > 60:
                            raise TimeoutError("Download stalled - no progress for 60 seconds")
                        
                        last_progress_time = current_time
                        
                        # Progress logging
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (20 * 1024 * 1024) == 0:  # Log every 20MB
                                print(f"Download progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)", file=sys.stderr)
            
            file_size = os.path.getsize(model_path) / (1024*1024)
            print(f"✓ Model downloaded successfully: {file_size:.1f}MB", file=sys.stderr)
            
        except Exception as e:
            print(f"✗ Model download failed: {str(e)}", file=sys.stderr)
            if os.path.exists(model_path):
                os.remove(model_path)
            raise
    else:
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"✓ Model file already exists: {file_size:.1f}MB", file=sys.stderr)

_model_cache = None

def model_fn(model_dir="model"):
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    try:
        device = torch.device("cpu")
        print(f"Using device: {device}", file=sys.stderr)
        
        # Create model without loading heavy components
        print("Creating lightweight model for testing...", file=sys.stderr)
        
        # Skip BERT entirely for now
        print("⚠️ SKIPPING BERT MODEL (testing mode)", file=sys.stderr)
        
        # Use dummy tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Use tiny Whisper
        print("Loading minimal Whisper model...", file=sys.stderr)
        transcriber = whisper.load_model("tiny", device="cpu")
        
        # Create dummy model
        class DummyModel:
            def __init__(self):
                self.device = device
            def eval(self):
                pass
            def __call__(self, text_inputs, video_frames, audio_features):
                # Return dummy predictions
                batch_size = text_inputs['input_ids'].shape[0]
                return {
                    'emotions': torch.randn(batch_size, 7),  # 7 emotions
                    'sentiments': torch.randn(batch_size, 3)  # 3 sentiments
                }
        
        model = DummyModel()
        
        _model_cache = {
            'model': model,
            'tokenizer': tokenizer,
            'transcriber': transcriber,
            'device': device
        }
        
        print("✓ Lightweight models loaded successfully", file=sys.stderr)
        return _model_cache
        
    except Exception as e:
        print(f"✗ Error: {str(e)}", file=sys.stderr)
        raise

def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']

    print(f"Starting transcription of: {video_path}", file=sys.stderr)
    log_memory_usage("before_transcription")
    
    try:
        # Add timeout for transcription
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Transcription timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2 minute timeout
        
        try:
            result = model_dict['transcriber'].transcribe(
                video_path, word_timestamps=True)
            print(f"✓ Transcription completed. Found {len(result['segments'])} segments", file=sys.stderr)
        finally:
            signal.alarm(0)
            
    except Exception as e:
        print(f"✗ Transcription failed: {str(e)}", file=sys.stderr)
        raise
    
    log_memory_usage("after_transcription")

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    print(f"Processing {len(result['segments'])} segments...", file=sys.stderr)
    
    for i, segment in enumerate(result["segments"]):
        segment_path = None
        print(f"Processing segment {i+1}/{len(result['segments'])}: {segment['start']:.1f}s-{segment['end']:.1f}s", file=sys.stderr)
        
        try:
            # Add debug logging for each step
            print(f"  Extracting segment...", file=sys.stderr)
            segment_path = utterance_processor.extract_segment(
                video_path,
                segment["start"],
                segment["end"]
            )
            print(f"  ✓ Segment extracted: {segment_path}", file=sys.stderr)

            print(f"  Processing video frames...", file=sys.stderr)
            video_frames = utterance_processor.video_processor.process_video(segment_path)
            print(f"  ✓ Video frames processed: {video_frames.shape}", file=sys.stderr)
            
            print(f"  Processing audio features...", file=sys.stderr)
            audio_features = utterance_processor.audio_processor.extract_features(segment_path)
            print(f"  ✓ Audio features processed: {audio_features.shape}", file=sys.stderr)
            
            print(f"  Tokenizing text: '{segment['text'][:50]}...'", file=sys.stderr)
            text_inputs = tokenizer(
                segment["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            print(f"  ✓ Text tokenized", file=sys.stderr)

            # Move to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            print(f"  Running model inference...", file=sys.stderr)
            # Get predictions
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)
                
            print(f"  ✓ Inference completed", file=sys.stderr)

            predictions.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "emotions": [
                    {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()} 
                    for idx, conf in zip(emotion_indices, emotion_values)
                ],
                "sentiments": [
                    {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()} 
                    for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            })
            print(f"✓ Segment {i+1} processed successfully", file=sys.stderr)

        except Exception as e:
            print(f"✗ Segment {i+1} failed: {str(e)}", file=sys.stderr)
            import traceback
            print(f"  Traceback: {traceback.format_exc()}", file=sys.stderr)

        finally:
            # Cleanup segment file
            if segment_path and os.path.exists(segment_path):
                os.remove(segment_path)
                
    print(f"✓ All segments processed. Total predictions: {len(predictions)}", file=sys.stderr)
    return {"utterances": predictions}


def process_video_from_url(video_url, model_dir="model"):
    """Process video from YouTube URL"""
    downloader = YouTubeDownloader()
    video_path = None
    
    try:
        print("=== Starting Video Analysis ===", file=sys.stderr)
        log_memory_usage("start_analysis")
        
        # Step 1: Download video
        print("Step 1/4: Downloading video...", file=sys.stderr)
        video_path = downloader.download_video(video_url)
        print(f"✓ Video downloaded: {video_path}", file=sys.stderr)
        log_memory_usage("after_download")
        
        # Step 2: Load model (cached after first call)
        print("Step 2/4: Loading ML models...", file=sys.stderr)
        model_dict = model_fn(model_dir)
        print("✓ Models ready", file=sys.stderr)
        log_memory_usage("after_models_ready")
        
        # Step 3: Process video
        print("Step 3/4: Processing video and running inference...", file=sys.stderr)
        input_data = {'video_path': video_path}
        log_memory_usage("before_inference")
        
        predictions = predict_fn(input_data, model_dict)
        log_memory_usage("after_inference")
        print("✓ Video processing complete", file=sys.stderr)
        
        # Step 4: Cleanup
        print("Step 4/4: Cleaning up temporary files...", file=sys.stderr)
        return predictions
        
    except Exception as e:
        print(f"✗ Error in video analysis: {str(e)}", file=sys.stderr)
        log_memory_usage("after_error")
        return {"error": str(e)}
        
    finally:
        # Cleanup downloaded video
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"✓ Cleaned up: {video_path}", file=sys.stderr)
        log_memory_usage("final_cleanup")


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
