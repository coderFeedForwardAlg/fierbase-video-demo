import React from 'react';
import { fireEvent, waitFor, screen } from '@testing-library/react';
import { render } from '../../../test-utils';
import VideoPage from '../../../components/pages/VideoPage';
import { getAuth } from 'firebase/auth';
import { getStorage, ref, listAll, getDownloadURL } from 'firebase/storage';

// Mock useNavigate and useLocation
const mockNavigate = jest.fn();
const mockLocation = {
  state: { projectName: 'test-project' }
};

// Override the mock from test-utils.js
jest.spyOn(require('react-router-dom'), 'useNavigate').mockImplementation(() => mockNavigate);
jest.spyOn(require('react-router-dom'), 'useLocation').mockImplementation(() => mockLocation);

describe('VideoPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock the auth.currentUser
    getAuth.mockReturnValue({
      currentUser: { uid: 'test-user-id' }
    });
  });

  test('renders loading state initially', () => {
    // Mock the storage functions for loading state
    listAll.mockReturnValue(new Promise(() => {})); // Never resolves to keep loading
    
    render(<VideoPage />);
    
    expect(screen.getByText('Video Player')).toBeInTheDocument();
    expect(screen.getByText('Videos in test-project')).toBeInTheDocument();
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  test('renders empty state when no videos exist', async () => {
    // Mock empty videos list
    listAll.mockResolvedValue({ items: [] });
    
    render(<VideoPage />);
    
    await waitFor(() => {
      expect(screen.getByText('No videos found in this project. Upload videos from the dashboard.')).toBeInTheDocument();
    });
  });

  test('renders videos list when videos exist', async () => {
    // Mock videos list
    const mockVideos = [
      { name: 'video1.mp4', fullPath: 'test-user-id/test-project/video1.mp4' },
      { name: 'video2.mp4', fullPath: 'test-user-id/test-project/video2.mp4' }
    ];
    
    listAll.mockResolvedValue({ items: mockVideos });
    
    getDownloadURL.mockImplementation((videoRef) => {
      if (videoRef.name === 'video1.mp4') {
        return Promise.resolve('https://example.com/video1.mp4');
      } else if (videoRef.name === 'video2.mp4') {
        return Promise.resolve('https://example.com/video2.mp4');
      }
      return Promise.resolve('');
    });
    
    render(<VideoPage />);
    
    await waitFor(() => {
      expect(screen.getByText('video1.mp4')).toBeInTheDocument();
      expect(screen.getByText('video2.mp4')).toBeInTheDocument();
    });
  });

  test('plays selected video when clicked', async () => {
    // Mock videos list
    const mockVideos = [
      { name: 'video1.mp4', fullPath: 'test-user-id/test-project/video1.mp4' }
    ];
    
    listAll.mockResolvedValue({ items: mockVideos });
    
    getDownloadURL.mockResolvedValue('https://example.com/video1.mp4');
    
    render(<VideoPage />);
    
    await waitFor(() => {
      expect(screen.getByText('video1.mp4')).toBeInTheDocument();
    });
    
    // Click on the video
    fireEvent.click(screen.getByText('video1.mp4'));
    
    await waitFor(() => {
      expect(screen.getByText('Now Playing')).toBeInTheDocument();
      expect(screen.getByText('Project: test-project')).toBeInTheDocument();
      
      // Check if video player has the correct source
      const videoPlayer = screen.getByTestId('video-player');
      expect(videoPlayer.src).toBe('https://example.com/video1.mp4');
    });
  });

  test('navigates to projects page when back button is clicked', async () => {
    listAll.mockResolvedValue({ items: [] });
    
    render(<VideoPage />);
    
    // Click back button
    fireEvent.click(screen.getByText('Back to Projects'));
    
    expect(mockNavigate).toHaveBeenCalledWith('/projects');
  });

  test('navigates to upload page when upload button is clicked', async () => {
    listAll.mockResolvedValue({ items: [] });
    
    render(<VideoPage />);
    
    // Click upload button
    fireEvent.click(screen.getByText('Upload New Video'));
    
    expect(mockNavigate).toHaveBeenCalledWith('/upload-project', { state: { projectName: 'test-project' } });
  });

  test('uses default project when no project name is provided', async () => {
    // Override the mock location to simulate no project name in state
    mockLocation.state = null;
    
    listAll.mockResolvedValue({ items: [] });
    
    render(<VideoPage />);
    
    await waitFor(() => {
      expect(screen.getByText('Videos in default_project')).toBeInTheDocument();
    });
  });
});
