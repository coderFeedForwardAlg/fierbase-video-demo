import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { getAuth, signOut } from 'firebase/auth';
import { getStorage, ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';
import { v4 as uuidv4 } from 'uuid';
import { toast } from 'react-toastify';
import './SecondPage.css';

function UploadPage() {
  const auth = getAuth();
  const user = auth.currentUser;
  const navigate = useNavigate();
  const location = useLocation();
  const [videoPreview, setVideoPreview] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [projectName, setProjectName] = useState(location.state?.projectName || '');
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const uploadVideo = async (file) => {
    if (!file) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      const storage = getStorage();
      const fileExtension = file.name.split('.').pop();
      const fileName = `${uuidv4()}.${fileExtension}`;
      const storagePath = `${user.uid}/${projectName}/${fileName}`;
      const storageRef = ref(storage, storagePath);
      
      // Get the MIME type from the file
      const fileType = file.type || 'video/mp4'; // Default to mp4 if type not available
      
      // Create file metadata including the content type
      const metadata = {
        contentType: fileType
      };
      
      const uploadTask = uploadBytesResumable(storageRef, file, metadata);
      
      uploadTask.on('state_changed',
        (snapshot) => {
          // Track upload progress
          const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
          setUploadProgress(progress);
        },
        (error) => {
          console.error('Upload failed:', error);
          toast.error('Upload failed. Please try again.');
          setIsUploading(false);
        },
        async () => {
          // Upload completed successfully
          const downloadURL = await getDownloadURL(uploadTask.snapshot.ref);
          toast.success('Video uploaded successfully!');
          console.log('File available at', downloadURL);
          
          // Reset the file input
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
          
          setUploadProgress(0);
          setIsUploading(false);
          setVideoPreview(null);
          
          // Navigate to the video page for this project
          navigate('/video-page', { state: { projectName } });
        }
      );
      
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error('Error uploading file. Please try again.');
      setIsUploading(false);
    }
  };

  const handleCancel = () => {
    if (location.state?.projectName) {
      // If we came from a specific project, go back to that project
      navigate('/video-page', { state: { projectName: location.state.projectName } });
    } else {
      // Otherwise go back to projects list
      navigate('/projects');
    }
  };

  if (!user) {
    return null; // or a loading spinner
  }

  return (
    <div className="second-page">
      <div className="user-card">
        <h1>{projectName ? `Upload to ${projectName}` : 'Create New Project'}</h1>

        <div className="video-upload-section">
          <div className="project-input-container">
            <div className="project-input">
              <label htmlFor="project-name">Project Name:</label>
              <input
                id="project-name"
                type="text"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="Enter project name"
                disabled={isUploading}
                className="project-name-input"
              />
              <p className="input-help">Give your project a descriptive name</p>
            </div>
          </div>
          
          <h2>Upload a Video</h2>
          <div className="file-upload-container">
            <input 
              type="file" 
              id="video-upload" 
              ref={fileInputRef}
              accept="video/*" 
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) {
                  // Check file size (100MB limit)
                  if (file.size > 100 * 1024 * 1024) {
                    toast.error('File size too large. Maximum size is 100MB.');
                    return;
                  }
                  
                  toast.info(`Selected file: ${file.name}`);
                  // Create a preview URL for the video
                  const videoUrl = URL.createObjectURL(file);
                  setVideoPreview({
                    url: videoUrl,
                    file: file,
                    name: file.name
                  });
                }
              }}
              style={{ display: 'none' }}
              disabled={isUploading}
            />
            <label htmlFor="video-upload" className="upload-button">
              Choose Video File
            </label>
            <p className="file-info">MP4, WebM, or MOV. Max 100MB</p>
            
            {isUploading && (
              <div className="upload-progress">
                <div 
                  className="progress-bar" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
                <div className="progress-text">
                  Uploading: {Math.round(uploadProgress)}%
                </div>
              </div>
            )}
            
            {videoPreview && !isUploading && (
              <div className="video-preview">
                <h3>Video Preview</h3>
                <video 
                  ref={videoRef}
                  src={videoPreview.url} 
                  controls 
                  className="preview-video"
                  onLoad={() => {
                    // Clean up the object URL when component unmounts
                    return () => URL.revokeObjectURL(videoPreview.url);
                  }}
                />
                <div className="video-controls">
                  <button 
                    onClick={() => videoRef.current.play()}
                    className="control-button play-button"
                  >
                    Play
                  </button>
                  <button 
                    onClick={() => videoRef.current.pause()}
                    className="control-button pause-button"
                  >
                    Pause
                  </button>
                  <button 
                    onClick={() => uploadVideo(videoPreview.file)}
                    className="control-button upload-button"
                    disabled={!projectName.trim()}
                  >
                    {projectName.trim() ? 'Upload Video' : 'Enter Project Name First'}
                  </button>
                </div>
                <p className="file-details">{videoPreview.name}</p>
              </div>
            )}
          </div>
        </div>
        
        <div className="button-group">
          <button onClick={handleCancel} className="back-button">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
