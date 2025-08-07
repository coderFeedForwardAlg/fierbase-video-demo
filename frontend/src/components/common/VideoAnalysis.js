import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { analyzeVideo } from '../../utils/api';
import './VideoAnalysis.css';

const VideoAnalysis = ({ userId, projectName, videoName, videoUrl }) => {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const handleAnalyzeVideo = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await analyzeVideo(userId, projectName, videoName);
      console.log(result);
      setAnalysis(result.analysis);
      toast.success('Video analysis completed!');
    } catch (err) {
      console.error('Error analyzing video:', err);
      setError(err.message || 'Failed to analyze video');
      toast.error('Failed to analyze video. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const toggleExpanded = () => {
    setExpanded(!expanded);
  };

  const renderAnalysisContent = () => {
    if (!analysis) return null;

    // Handle raw analysis (text) case
    if (analysis.raw_analysis) {
      return <div className="analysis-raw">{analysis.raw_analysis}</div>;
    }

    // Handle structured JSON response
    return (
      <div className="analysis-sections">
        {analysis.description && (
          <div className="analysis-section">
            <h4>Description</h4>
            <p>{analysis.description}</p>
          </div>
        )}
        
        {analysis.keyObjects && (
          <div className="analysis-section">
            <h4>Key Objects/People</h4>
            {typeof analysis.keyObjects === 'string' ? (
              <p>{analysis.keyObjects}</p>
            ) : (
              <ul>
                {analysis.keyObjects.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        
        {analysis.notableEvents && (
          <div className="analysis-section">
            <h4>Notable Events</h4>
            {typeof analysis.notableEvents === 'string' ? (
              <p>{analysis.notableEvents}</p>
            ) : (
              <ul>
                {analysis.notableEvents.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        
        {analysis.suggestedEdits && (
          <div className="analysis-section">
            <h4>Suggested Edits</h4>
            {typeof analysis.suggestedEdits === 'string' ? (
              <p>{analysis.suggestedEdits}</p>
            ) : (
              <ul>
                {analysis.suggestedEdits.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="video-analysis">
      <div className="video-analysis-header">
        <button 
          className="analyze-button" 
          onClick={handleAnalyzeVideo} 
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze with AI'}
        </button>
        
        {analysis && (
          <button 
            className="toggle-button" 
            onClick={toggleExpanded}
          >
            {expanded ? 'Hide Analysis' : 'Show Analysis'}
          </button>
        )}
      </div>
      
      {error && <div className="analysis-error">{error}</div>}
      
      {analysis && expanded && (
        <div className="analysis-content">
          {renderAnalysisContent()}
        </div>
      )}
    </div>
  );
};

export default VideoAnalysis;
