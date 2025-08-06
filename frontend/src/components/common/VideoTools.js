import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { applyVideoTools } from '../../utils/api';
import './VideoTools.css';

const VideoTools = ({ userId, projectName, videoUrl }) => {
  const [loading, setLoading] = useState(false);
  const [instructions, setInstructions] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const handleApplyTools = async () => {
    if (!instructions.trim()) {
      toast.warning('Please enter instructions for the AI');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResults(null);
      
      const response = await applyVideoTools(userId, projectName, videoUrl, instructions);
      
      if (response.success) {
        setResults(response.results);
        setShowResults(true);
        toast.success('Video tools applied successfully!');
      } else {
        setError(response.error || 'Failed to apply video tools');
        toast.error('Failed to apply video tools');
      }
    } catch (err) {
      console.error('Error applying video tools:', err);
      setError(err.message || 'Failed to apply video tools');
      toast.error('Failed to apply video tools. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const renderToolResult = (result, index) => {
    const { tool, args, result: toolResult } = result;
    
    return (
      <div key={index} className="tool-result">
        <h4>{tool}</h4>
        <div className="tool-args">
          <strong>Parameters:</strong>
          <ul>
            {Object.entries(args).map(([key, value]) => (
              <li key={key}>
                <span className="arg-name">{key}:</span> {JSON.stringify(value)}
              </li>
            ))}
          </ul>
        </div>
        
        {toolResult.success ? (
          <div className="tool-success">
            <strong>Success!</strong>
            {toolResult.output_url && (
              <div className="tool-output">
                <p>Modified video:</p>
                <video 
                  className="result-video" 
                  controls
                  src={toolResult.output_url}
                >
                  Your browser does not support the video tag.
                </video>
              </div>
            )}
          </div>
        ) : (
          <div className="tool-error">
            <strong>Error:</strong> {toolResult.error}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="video-tools">
      <h3>AI Video Tools</h3>
      
      <div className="tools-form">
        <div className="form-group">
          <label htmlFor="instructions">Instructions for AI:</label>
          <textarea
            id="instructions"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="Enter instructions for the AI (e.g., 'trim this video to show only the first 10 seconds' or 'speed up this video by 2x')"
            disabled={loading}
            rows={3}
          />
        </div>
        
        <button 
          className="apply-tools-button" 
          onClick={handleApplyTools} 
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Apply AI Tools'}
        </button>
      </div>
      
      {error && (
        <div className="tools-error">
          {error}
        </div>
      )}
      
      {results && showResults && (
        <div className="tools-results">
          <h4>Results</h4>
          {results.length === 0 ? (
            <p>No tools were applied. Try different instructions.</p>
          ) : (
            results.map((result, index) => renderToolResult(result, index))
          )}
        </div>
      )}
    </div>
  );
};

export default VideoTools;
