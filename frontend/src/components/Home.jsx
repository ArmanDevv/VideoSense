import React, { useState } from 'react';
import { Search, Play, BarChart3, User, LogOut, Youtube, Sparkles, Clock, ThumbsUp, ThumbsDown, AlertCircle, Loader, X, ChevronRight } from 'lucide-react';


const HomePage = () => {
  const [companyName, setCompanyName] = useState('');
  const [productName, setProductName] = useState('');
  const [videos, setVideos] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzedVideos, setAnalyzedVideos] = useState([]);
  const [searchPerformed, setSearchPerformed] = useState(false);
  const [selectedVideoAnalysis, setSelectedVideoAnalysis] = useState(null);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  // ✅ NEW: Track which video is currently being analyzed
  const [analyzingVideoId, setAnalyzingVideoId] = useState(null);


  const user = {
    name: localStorage.getItem('userName') || 'Guest'
  };


  const YOUTUBE_API_KEY = import.meta.env.VITE_YOUTUBE_API;


  const parseDurationSeconds = (iso) => {
    const regex = /PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/;
    const [, h, m, s] = regex.exec(iso) || [];
    const hours = h ? parseInt(h) : 0;
    const mins = m ? parseInt(m) : 0;
    const secs = s ? parseInt(s) : 0;
    return hours * 3600 + mins * 60 + secs;
  };


  const formatDuration = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return h ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}` : `${m}:${s.toString().padStart(2, '0')}`;
  };


  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };


  const searchYouTubeVideos = async (company, product) => {
    setIsSearching(true);
    const query = `${company} ${product} review in English`;
    const searchUrl = `https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=5&q=${encodeURIComponent(query)}&key=${YOUTUBE_API_KEY}`;
    
    try {
      const searchResponse = await fetch(searchUrl);
      const searchData = await searchResponse.json();


      if (!searchData.items || searchData.items.length === 0) {
        setVideos([]);
        setSearchPerformed(true);
        setIsSearching(false);
        return;
      }


      const videoIds = searchData.items.map(item => item.id.videoId).join(',');
      const detailsUrl = `https://www.googleapis.com/youtube/v3/videos?part=contentDetails,statistics&id=${videoIds}&key=${YOUTUBE_API_KEY}`;
      const detailsResponse = await fetch(detailsUrl);
      const detailsData = await detailsResponse.json();


      const videos = searchData.items.map(item => {
        const details = detailsData.items.find(d => d.id === item.id.videoId);
        const durationSeconds = details ? parseDurationSeconds(details.contentDetails.duration) : 0;
        return {
          id: item.id.videoId,
          title: item.snippet.title,
          channel: item.snippet.channelTitle,
          thumbnail: item.snippet.thumbnails.high.url,
          url: `https://youtube.com/watch?v=${item.id.videoId}`,
          duration: formatDuration(durationSeconds),
          durationSeconds,
          views: details ? details.statistics.viewCount : 'N/A',
          publishedAt: new Date(item.snippet.publishedAt).toLocaleDateString()
        };
      }).filter(video => video.durationSeconds <= 60);


      setVideos(videos);
    } catch (error) {
      setVideos([]);
    }


    setSearchPerformed(true);
    setIsSearching(false);
  };


  const analyzeVideo = async (video) => {
    // ✅ Set loading state for this specific video
    setAnalyzingVideoId(video.id);
    
    try {
      const response = await fetch('https://videosense-production.up.railway.app/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoUrl: video.url }),
      });
      const result = await response.json();
      
      setAnalyzedVideos(prev => [
        ...prev.filter(v => v.id !== video.id),
        { ...video, analysis: result }
      ]);
      
      return result;
    } catch (error) {
      const errorResult = { utterances: [], error: 'Analysis failed' };
      setAnalyzedVideos(prev => [
        ...prev.filter(v => v.id !== video.id),
        { ...video, analysis: errorResult }
      ]);
      return errorResult;
    } finally {
      // ✅ Clear loading state when analysis completes (success or failure)
      setAnalyzingVideoId(null);
    }
  };


  const handleSearch = async (e) => {
    e.preventDefault();
    if (companyName.trim() && productName.trim()) {
      await searchYouTubeVideos(companyName, productName);
    }
  };


  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-100';
      case 'negative': return 'text-red-600 bg-red-100';
      default: return 'text-yellow-600 bg-yellow-100';
    }
  };


  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive': return <ThumbsUp className="w-4 h-4" />;
      case 'negative': return <ThumbsDown className="w-4 h-4" />;
      default: return <AlertCircle className="w-4 h-4" />;
    }
  };


  const getEmotionColor = (emotion) => {
    const colors = {
      sadness: 'text-blue-600 bg-blue-100',
      anger: 'text-red-600 bg-red-100',
      fear: 'text-purple-600 bg-purple-100',
      joy: 'text-yellow-600 bg-yellow-100',
      disgust: 'text-green-600 bg-green-100',
      surprise: 'text-pink-600 bg-pink-100',
      neutral: 'text-gray-600 bg-gray-100'
    };
    return colors[emotion] || 'text-gray-600 bg-gray-100';
  };


const calculateOverallSentiment = (utterances) => {
  if (!utterances || utterances.length === 0) {
    return { sentiment: 'neutral', confidence: 0, analysis: 'No content' };
  }
  
  let strongPositive = 0, strongNegative = 0;
  let weakPositive = 0, weakNegative = 0;
  let neutral = 0;
  
  const strongThreshold = 0.7;
  const weakThreshold = 0.5;
  
  utterances.forEach(utterance => {
    if (utterance.sentiments && utterance.sentiments.length > 0) {
      const sent = utterance.sentiments[0];
      const conf = sent.confidence;
      
      if (sent.label === 'positive') {
        if (conf >= strongThreshold) strongPositive++;
        else if (conf >= weakThreshold) weakPositive++;
      } else if (sent.label === 'negative') {
        if (conf >= strongThreshold) strongNegative++;
        else if (conf >= weakThreshold) weakNegative++;
      } else {
        neutral++;
      }
    }
  });
  
  // Decision logic
  const totalEmotional = strongPositive + strongNegative + weakPositive + weakNegative;
  const totalSegments = utterances.length;
  
  // If less than 20% is emotional content, it's neutral
  if (totalEmotional / totalSegments < 0.2) {
    return { 
      sentiment: 'neutral', 
      confidence: 60,
      analysis: `Mostly neutral content (${totalEmotional}/${totalSegments} emotional segments)`
    };
  }
  
  // Strong emotions override weak ones
  if (strongPositive > strongNegative) {
    return { 
      sentiment: 'positive', 
      confidence: 80,
      analysis: `${strongPositive} strong positive, ${strongNegative} strong negative`
    };
  } else if (strongNegative > strongPositive) {
    return { 
      sentiment: 'negative', 
      confidence: 80,
      analysis: `${strongNegative} strong negative, ${strongPositive} strong positive`
    };
  }
  
  // Fall back to weak sentiments
  const totalPositive = strongPositive + weakPositive;
  const totalNegative = strongNegative + weakNegative;
  
  if (totalPositive > totalNegative) {
    return { sentiment: 'positive', confidence: 65, analysis: 'Mostly positive' };
  } else if (totalNegative > totalPositive) {
    return { sentiment: 'negative', confidence: 65, analysis: 'Mostly negative' };
  } else {
    return { sentiment: 'neutral', confidence: 70, analysis: 'Mixed emotions' };
  }
};



  const showDetailedAnalysis = (video) => {
    setSelectedVideoAnalysis(video);
    setShowAnalysisModal(true);
  };


  const AnalysisModal = () => {
    if (!showAnalysisModal || !selectedVideoAnalysis) return null;


    const { analysis } = selectedVideoAnalysis;
    const overallSentiment = calculateOverallSentiment(analysis.utterances);


    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
          <div className="flex justify-between items-center p-6 border-b">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Detailed Analysis</h2>
              <p className="text-gray-600 text-sm mt-1 truncate">{selectedVideoAnalysis.title}</p>
            </div>
            <button 
              onClick={() => setShowAnalysisModal(false)}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>
          </div>


          <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            {/* Overall Summary */}
            <div className="bg-gray-50 rounded-xl p-4 mb-6">
              <h3 className="font-semibold text-gray-900 mb-3">Overall Sentiment</h3>
              <div className="flex items-center space-x-4">
                <div className={`flex items-center space-x-2 px-3 py-2 rounded-full text-sm font-medium ${getSentimentColor(overallSentiment.sentiment)}`}>
                  {getSentimentIcon(overallSentiment.sentiment)}
                  <span className="capitalize">{overallSentiment.sentiment}</span>
                </div>
                <span className="text-sm text-gray-600">
                  {overallSentiment.confidence}% confidence
                </span>
              </div>
            </div>


            {/* Time-segmented Analysis */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Time-segmented Analysis</h3>
              {analysis.error ? (
                <div className="text-red-600 text-center py-8">
                  <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                  <p>{analysis.error}</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {analysis.utterances?.map((utterance, index) => {
                    const primarySentiment = utterance.sentiments?.[0] || { label: 'neutral', confidence: 0 };
                    const primaryEmotion = utterance.emotions?.[0] || { label: 'neutral', confidence: 0 };
                    
                    return (
                      <div key={index} className="border border-gray-200 rounded-xl p-4">
                        <div className="flex justify-between items-start mb-3">
                          <div className="flex items-center space-x-2 text-sm text-gray-500">
                            <Clock className="w-4 h-4" />
                            <span>{formatTime(utterance.start_time)} - {formatTime(utterance.end_time)}</span>
                          </div>
                          <div className="flex space-x-2">
                            <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(primarySentiment.label)}`}>
                              {getSentimentIcon(primarySentiment.label)}
                              <span className="capitalize">{primarySentiment.label}</span>
                              <span>({Math.round(primarySentiment.confidence * 100)}%)</span>
                            </div>
                            <div className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${getEmotionColor(primaryEmotion.label)}`}>
                              {primaryEmotion.label} ({Math.round(primaryEmotion.confidence * 100)}%)
                            </div>
                          </div>
                        </div>
                        
                        <p className="text-gray-800 mb-3">{utterance.text}</p>
                        
                        {/* Detailed sentiment breakdown */}
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <p className="text-xs text-gray-600 font-medium mb-2">Sentiment Breakdown:</p>
                            <div className="space-y-1">
                              {utterance.sentiments?.map((sent, i) => (
                                <div key={i} className="flex justify-between text-xs">
                                  <span className="capitalize">{sent.label}:</span>
                                  <span>{Math.round(sent.confidence * 100)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <p className="text-xs text-gray-600 font-medium mb-2">Emotion Breakdown:</p>
                            <div className="space-y-1">
                              {utterance.emotions?.slice(0, 3).map((emotion, i) => (
                                <div key={i} className="flex justify-between text-xs">
                                  <span className="capitalize">{emotion.label}:</span>
                                  <span>{Math.round(emotion.confidence * 100)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };


  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
                <Play className="w-4 h-4 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">VideoSense</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-gray-600">
                <User className="w-4 h-4" />
                <span className="font-medium">{user.name}</span>
              </div>
              <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>


      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome back, {user.name}!
          </h1>
          <p className="text-gray-600">
            Search for YouTube reviews of your products and get instant AI-powered sentiment analysis.
          </p>
        </div>


        {/* Search Form */}
        <div className="bg-white rounded-2xl shadow-sm border p-8 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
            <Search className="w-5 h-5 mr-2" />
            Find Product Reviews
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Company Name
              </label>
              <input
                type="text"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="e.g. Apple, Samsung, Nike..."
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Product Name
              </label>
              <input
                type="text"
                value={productName}
                onChange={(e) => setProductName(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="e.g. iPhone 15, Galaxy S24, Air Max..."
              />
            </div>
          </div>
          
          <button
            onClick={handleSearch}
            disabled={isSearching || !companyName.trim() || !productName.trim()}
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center space-x-2"
          >
            {isSearching ? (
              <>
                <Loader className="w-4 h-4 animate-spin" />
                <span>Searching YouTube...</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                <span>Search Videos</span>
              </>
            )}
          </button>
        </div>


        {/* Search Results */}
        {searchPerformed && (
          <div className="bg-white rounded-2xl shadow-sm border p-8 mb-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Youtube className="w-5 h-5 mr-2 text-red-500" />
                Found {videos.length} Reviews
              </h2>
            </div>


            <div className="grid gap-6">
              {videos.map((video) => {
                const analyzed = analyzedVideos.find(v => v.id === video.id);
                const overallSentiment = analyzed ? calculateOverallSentiment(analyzed.analysis.utterances) : null;
                // ✅ Check if this video is currently being analyzed
                const isCurrentlyAnalyzing = analyzingVideoId === video.id;
                
                return (
                  <div key={video.id} className="border border-gray-200 rounded-xl p-6 hover:border-blue-300 transition-colors">
                    <div className="flex space-x-4">
                      <div className="flex-shrink-0">
                        <div className="w-32 h-20 bg-gray-200 rounded-lg flex items-center justify-center">
                          <Play className="w-8 h-8 text-gray-400" />
                        </div>
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
                          {video.title}
                        </h3>
                        <div className="flex items-center space-x-4 text-sm text-gray-500 mb-3">
                          <span>{video.channel}</span>
                          <span>•</span>
                          <span>{video.views} views</span>
                          <span>•</span>
                          <span>{video.publishedAt}</span>
                          <span>•</span>
                          <div className="flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>{video.duration}</span>
                          </div>
                        </div>
                        
                        <a 
                          href={video.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                        >
                          Watch on YouTube →
                        </a>


                        {/* Overall sentiment display */}
                        {analyzed && overallSentiment && (
                          <div className="mt-3">
                            <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(overallSentiment.sentiment)}`}>
                              {getSentimentIcon(overallSentiment.sentiment)}
                              <span className="capitalize">{overallSentiment.sentiment}</span>
                              <span>({overallSentiment.confidence}%)</span>
                            </div>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex-shrink-0 w-48">
                        {/* ✅ Show spinner when analyzing this specific video */}
                        {isCurrentlyAnalyzing ? (
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex flex-col items-center justify-center space-y-2">
                            <Loader className="w-6 h-6 animate-spin text-blue-500" />
                            <span className="text-blue-700 font-medium text-sm">Analyzing </span>
                            <span className="text-blue-600 text-xs text-center">
                              Takes about a min
                            </span>
                          </div>
                        ) : analyzed ? (
                          <div className="space-y-2">
                            <button
                              onClick={() => showDetailedAnalysis(analyzed)}
                              className="w-full px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2"
                            >
                              <BarChart3 className="w-4 h-4" />
                              <span>View Analysis</span>
                              <ChevronRight className="w-4 h-4" />
                            </button>
                            <p className="text-xs text-gray-500 text-center">
                              {analyzed.analysis.utterances?.length || 0} segments analyzed
                            </p>
                          </div>
                        ) : (
                          <button
                            onClick={() => analyzeVideo(video)}
                            disabled={analyzingVideoId !== null} // ✅ Disable all analyze buttons when any video is analyzing
                            className="w-full px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center space-x-2"
                          >
                            <Sparkles className="w-4 h-4" />
                            <span>Analyze</span>
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}


        {/* Empty State */}
        {!searchPerformed && (
          <div className="bg-white rounded-2xl shadow-sm border p-12 text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Ready to Analyze Reviews?
            </h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Enter your company and product name above to find YouTube reviews and get instant AI-powered sentiment analysis.
            </p>
          </div>
        )}
      </div>


      {/* Analysis Modal */}
      <AnalysisModal />
    </div>
  );
};


export default HomePage;
