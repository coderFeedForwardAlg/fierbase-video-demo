/**
 * API utility functions for interacting with the Video Vault API
 */

// Replace this with your actual Cloud Run URL after deployment
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://genkit-container-hwfwgzu75q-ew.a.run.app/';

/**
 * Generic fetch wrapper with error handling
 * @param {string} endpoint - API endpoint to call
 * @param {Object} options - Fetch options
 * @returns {Promise<any>} - Response data
 */
export const fetchApi = async (endpoint, options = {}) => {
  try {
    // Ensure API_BASE_URL ends with a slash if endpoint doesn't start with one
    const baseUrl = API_BASE_URL.endsWith('/') || endpoint.startsWith('/') ? API_BASE_URL : `${API_BASE_URL}/`;
    const url = `${baseUrl}${endpoint.startsWith('/') ? endpoint.substring(1) : endpoint}`;
    
    console.log('Fetching from URL:', url); // Debug log
    
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
  const response = await fetch('https://genkit-container-656666497922.europe-west1.run.app/api/analyze-video', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ userId, projectName, videoName }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  
  return await response.json();
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

