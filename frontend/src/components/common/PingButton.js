import React, { useState } from 'react';
import { pingApi } from '../../utils/api';

/**
 * A button component that pings the backend API
 */
const PingButton = () => {
  const [pingResult, setPingResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePingClick = async () => {
    try {
      setLoading(true);
      setError(null);
      setPingResult(null);
      
      const result = await pingApi();
      
      setPingResult(result);
    } catch (err) {
      console.error('Error pinging API:', err);
      setError(err.message || 'Failed to ping API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ping-button-container" style={{ margin: '20px', padding: '20px', border: '1px solid #eee', borderRadius: '8px' }}>
      <h3>API Ping Test</h3>
      <button 
        onClick={handlePingClick}
        disabled={loading}
        style={{
          padding: '10px 20px',
          backgroundColor: '#4285F4',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '16px'
        }}
      >
        {loading ? 'Pinging...' : 'Ping API'}
      </button>
      
      {loading && <p>Loading...</p>}
      
      {error && (
        <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#ffebee', color: '#d32f2f', borderRadius: '4px' }}>
          Error: {error}
        </div>
      )}
      
      {pingResult && (
        <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#e8f5e9', color: '#2e7d32', borderRadius: '4px' }}>
          <h4>API Response:</h4>
          <pre style={{ whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(pingResult, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export default PingButton;
