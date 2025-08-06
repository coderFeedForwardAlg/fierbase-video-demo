/**
 * API utility functions for interacting with the Video Vault API
 */

// Replace this with your actual Cloud Run URL after deployment
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

/**
 * Generic fetch wrapper with error handling
 * @param {string} endpoint - API endpoint to call
 * @param {Object} options - Fetch options
 * @returns {Promise<any>} - Response data
 */
export const fetchApi = async (endpoint, options = {}) => {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    
    // Default options
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
      },
    };
    
    const response = await fetch(url, { ...defaultOptions, ...options });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
};

/**
 * Test the API connection with a ping
 * @returns {Promise<Object>} - Ping response
 */
export const pingApi = async () => {
  return await fetchApi('/ping');
};

/**
 * Check if the Genkit service is available
 * @returns {Promise<Object>} - Status response
 */
export const checkGenkitStatus = async () => {
  return await fetchApi('/api/genkit-status');
};

/**
 * Analyze a video using Genkit and Gemini 1.5
 * @param {string} userId - Firebase user ID
 * @param {string} projectName - Project name
 * @param {string} videoName - Video filename
 * @returns {Promise<Object>} - Analysis response
 */
export const analyzeVideo = async (userId, projectName, videoName) => {
  return await fetchApi('/api/analyze-video', {
    method: 'POST',
    body: JSON.stringify({ userId, projectName, videoName }),
  });
};

/**
 * Apply AI-powered tools to a video based on instructions
 * @param {string} userId - Firebase user ID
 * @param {string} projectName - Project name
 * @param {string} videoUrl - URL of the video
 * @param {string} toolInstructions - Instructions for the AI
 * @returns {Promise<Object>} - Tool results response
 */
export const applyVideoTools = async (userId, projectName, videoUrl, toolInstructions) => {
  return await fetchApi('/api/video-tools', {
    method: 'POST',
    body: JSON.stringify({ userId, projectName, videoUrl, toolInstructions }),
  });
};

