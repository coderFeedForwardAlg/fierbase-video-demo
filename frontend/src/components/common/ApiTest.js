import React, { useState, useEffect } from 'react';
import { pingApi } from '../../utils/api';

/**
 * Component to test API connectivity
 */
const ApiTest = () => {
  const [apiStatus, setApiStatus] = useState({
    loading: true,
    success: false,
    message: 'Testing API connection...',
    error: null
  });

  useEffect(() => {
    const testApiConnection = async () => {
      try {
        setApiStatus({
          loading: true,
          success: false,
          message: 'Testing API connection...',
          error: null
        });
        
        const response = await pingApi();
        
        setApiStatus({
          loading: false,
          success: true,
          message: `API connection successful: ${response.message}`,
          error: null
        });
      } catch (error) {
        setApiStatus({
          loading: false,
          success: false,
          message: 'API connection failed',
          error: error.message
        });
      }
    };

    testApiConnection();
  }, []);

  return (
    <div className="api-test">
      <h3>API Connection Status</h3>
      {apiStatus.loading ? (
        <p>Loading...</p>
      ) : apiStatus.success ? (
        <p style={{ color: 'green' }}>{apiStatus.message}</p>
      ) : (
        <div>
          <p style={{ color: 'red' }}>{apiStatus.message}</p>
          {apiStatus.error && <p>Error: {apiStatus.error}</p>}
        </div>
      )}
    </div>
  );
};

export default ApiTest;
