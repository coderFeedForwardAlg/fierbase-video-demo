import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { getAuth } from 'firebase/auth';
import { getStorage, ref, listAll, getDownloadURL } from 'firebase/storage';
import { toast } from 'react-toastify';
import './VideoPage.css';
import './SecondPage.css'; // For common styles

function VideoPage() {
  const auth = getAuth();
  const user = auth.currentUser;
  const navigate = useNavigate();
  const location = useLocation();
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [projectName, setProjectName] = useState(location.state?.projectName || 'default_project');

  // Fetch videos for the specific project
  useEffect(() => {
    if (!user || !projectName) return;
    
    const fetchVideos = async () => {
      try {
        setLoading(true);
        const storage = getStorage();
        const projectRef = ref(storage, `${user.uid}/${projectName}`);
        
        // List all videos in the project
        const videosResult = await listAll(projectRef);
        
        // Get metadata for each video
        const videosData = await Promise.all(
          videosResult.items.map(async (videoRef) => {
            const downloadURL = await getDownloadURL(videoRef);
            return {
              name: videoRef.name,
              path: videoRef.fullPath,
              url: downloadURL
            };
          })
        );
        
        setVideos(videosData);
      } catch (error) {
        console.error('Error fetching videos:', error);
        toast.error('Failed to load videos. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchVideos();
  }, [user, projectName]);
  
  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
  };

  if (!user) {
    return null; // or a loading spinner
  }

  return (
    <div className="second-page">
      <div className="user-card">
        <h1>Video Player</h1>
        
        <div className="video-container">
          <video 
            className="video-player" 
            controls
            src={selectedVideo?.url}
            poster={selectedVideo ? undefined : '/video-placeholder.png'}
          >
            Your browser does not support the video tag.
          </video>
          
          {selectedVideo && (
            <div className="video-info">
              <h3>Now Playing</h3>
              <p>{selectedVideo.name}</p>
              <p>Project: {projectName}</p>
            </div>
          )}
        </div>
        
        <div className="videos-container">
          <h2>Videos in {projectName}</h2>
          
          {loading ? (
            <div className="loading-spinner"></div>
          ) : videos.length === 0 ? (
            <p>No videos found in this project. Upload videos from the dashboard.</p>
          ) : (
            <div className="videos-list-container">
              <ul className="videos-list">
                {videos.map((video) => (
                  <li 
                    key={video.path} 
                    className={`video-item ${selectedVideo?.path === video.path ? 'selected' : ''}`}
                    onClick={() => handleVideoSelect(video)}
                  >
                    {video.name}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        
        <div className="button-group">
          <button onClick={() => navigate('/projects')} className="back-button">
            Back to Projects
          </button>
          <button onClick={() => navigate('/upload-project', { state: { projectName } })} className="action-button">
            Upload New Video
          </button>
        </div>
      </div>
    </div>
  );
}

export default VideoPage;
