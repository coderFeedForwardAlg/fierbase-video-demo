import React from 'react';
import { fireEvent, waitFor, screen, act } from '@testing-library/react';
import { render } from '../../test-utils';
import App from '../../App';
import { getAuth, signInWithEmailAndPassword } from 'firebase/auth';
import { getStorage, ref, listAll, getDownloadURL, uploadBytesResumable } from 'firebase/storage';

// Mock BrowserRouter and Routes
jest.mock('react-router-dom', () => {
  const originalModule = jest.requireActual('react-router-dom');
  return {
    ...originalModule,
    BrowserRouter: ({ children }) => <div>{children}</div>,
  };
});

// Mock UUID
jest.mock('uuid', () => ({
  v4: jest.fn(() => 'mock-uuid')
}));

describe('User Flow Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock URL.createObjectURL and URL.revokeObjectURL
    global.URL.createObjectURL = jest.fn(() => 'mock-url');
    global.URL.revokeObjectURL = jest.fn();
    
    // Setup auth mock to simulate login
    getAuth.mockReturnValue({
      currentUser: null,
      onAuthStateChanged: jest.fn((auth, callback) => {
        // Initially no user
        callback(null);
        // Return unsubscribe function
        return jest.fn();
      })
    });
  });

  test('Complete user flow: login -> select project -> play video', async () => {
    // Mock successful login
    signInWithEmailAndPassword.mockResolvedValueOnce({ 
      user: { uid: 'test-user-id' } 
    });
    
    // Mock auth state change after login
    getAuth.mockImplementation(() => ({
      currentUser: { uid: 'test-user-id' },
      onAuthStateChanged: jest.fn((auth, callback) => {
        // Simulate auth state change after login
        setTimeout(() => callback({ uid: 'test-user-id' }), 10);
        return jest.fn();
      })
    }));
    
    // Mock projects list
    const mockProjects = [
      { name: 'project1' }
    ];
    
    // Mock videos list
    const mockVideos = [
      { name: 'video1.mp4', fullPath: 'test-user-id/project1/video1.mp4' }
    ];
    
    // Setup storage mocks
    listAll.mockImplementation((ref) => {
      if (ref.fullPath === 'test-user-id') {
        return Promise.resolve({ prefixes: mockProjects });
      } else if (ref.fullPath === 'test-user-id/project1') {
        return Promise.resolve({ items: mockVideos });
      }
      return Promise.resolve({ items: [], prefixes: [] });
    });
    
    ref.mockImplementation((storage, path) => {
      return { fullPath: path, name: path.split('/').pop() };
    });
    
    getDownloadURL.mockResolvedValue('https://example.com/video1.mp4');
    
    // Render the app
    render(<App />);
    
    // Check if we're on the login page
    expect(screen.getByText('Welcome to Video Vault')).toBeInTheDocument();
    
    // Fill in login form
    fireEvent.change(screen.getByPlaceholderText('Email'), { 
      target: { value: 'test@example.com' } 
    });
    
    fireEvent.change(screen.getByPlaceholderText('Password'), { 
      target: { value: 'password123' } 
    });
    
    // Submit login form
    fireEvent.click(screen.getByText('Sign In'));
    
    // Check if login function was called
    expect(signInWithEmailAndPassword).toHaveBeenCalledWith(
      expect.anything(),
      'test@example.com',
      'password123'
    );
    
    // Wait for projects page to load after auth state change
    await waitFor(() => {
      expect(screen.getByText('Your Projects')).toBeInTheDocument();
    });
    
    // Check if projects are displayed
    await waitFor(() => {
      expect(screen.getByText('project1')).toBeInTheDocument();
    });
    
    // Click on project
    fireEvent.click(screen.getByText('project1'));
    
    // Wait for video page to load
    await waitFor(() => {
      expect(screen.getByText('Video Player')).toBeInTheDocument();
      expect(screen.getByText('Videos in project1')).toBeInTheDocument();
    });
    
    // Check if videos are displayed
    await waitFor(() => {
      expect(screen.getByText('video1.mp4')).toBeInTheDocument();
    });
    
    // Click on video
    fireEvent.click(screen.getByText('video1.mp4'));
    
    // Check if video player shows the video
    await waitFor(() => {
      expect(screen.getByText('Now Playing')).toBeInTheDocument();
      const videoPlayer = screen.getByTestId('video-player');
      expect(videoPlayer.src).toBe('https://example.com/video1.mp4');
    });
  });

  test('Complete user flow: login -> create project -> upload video', async () => {
    // Mock successful login
    signInWithEmailAndPassword.mockResolvedValueOnce({ 
      user: { uid: 'test-user-id' } 
    });
    
    // Mock auth state change after login
    getAuth.mockImplementation(() => ({
      currentUser: { uid: 'test-user-id' },
      onAuthStateChanged: jest.fn((auth, callback) => {
        // Simulate auth state change after login
        setTimeout(() => callback({ uid: 'test-user-id' }), 10);
        return jest.fn();
      })
    }));
    
    // Mock empty projects list
    listAll.mockImplementation((ref) => {
      if (ref.fullPath === 'test-user-id') {
        return Promise.resolve({ prefixes: [] });
      }
      return Promise.resolve({ items: [], prefixes: [] });
    });
    
    ref.mockImplementation((storage, path) => {
      return { fullPath: path, name: path.split('/').pop() };
    });
    
    // Mock upload task
    const mockUploadTask = {
      on: jest.fn((event, progressCallback, errorCallback, completeCallback) => {
        // Simulate upload progress
        progressCallback({ bytesTransferred: 50, totalBytes: 100 });
        // Simulate upload complete
        completeCallback();
        return mockUploadTask;
      }),
      snapshot: {
        ref: 'mock-ref'
      }
    };
    
    uploadBytesResumable.mockReturnValue(mockUploadTask);
    getDownloadURL.mockResolvedValue('https://example.com/uploaded-video.mp4');
    
    // Render the app
    render(<App />);
    
    // Check if we're on the login page
    expect(screen.getByText('Welcome to Video Vault')).toBeInTheDocument();
    
    // Fill in login form
    fireEvent.change(screen.getByPlaceholderText('Email'), { 
      target: { value: 'test@example.com' } 
    });
    
    fireEvent.change(screen.getByPlaceholderText('Password'), { 
      target: { value: 'password123' } 
    });
    
    // Submit login form
    fireEvent.click(screen.getByText('Sign In'));
    
    // Wait for projects page to load after auth state change
    await waitFor(() => {
      expect(screen.getByText('Your Projects')).toBeInTheDocument();
    });
    
    // Check if empty state is displayed
    await waitFor(() => {
      expect(screen.getByText("You don't have any projects yet.")).toBeInTheDocument();
    });
    
    // Click create project button
    fireEvent.click(screen.getByText('Create Your First Project'));
    
    // Wait for upload page to load
    await waitFor(() => {
      expect(screen.getByText('Create New Project')).toBeInTheDocument();
    });
    
    // Enter project name
    fireEvent.change(screen.getByLabelText('Project Name:'), { 
      target: { value: 'new-project' } 
    });
    
    // Upload a video
    const file = new File(['dummy content'], 'test-video.mp4', { type: 'video/mp4' });
    const fileInput = screen.getByLabelText('Choose Video File');
    
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    
    fireEvent.change(fileInput);
    
    // Wait for video preview
    await waitFor(() => {
      expect(screen.getByText('Video Preview')).toBeInTheDocument();
    });
    
    // Click upload button
    fireEvent.click(screen.getByText('Upload Video'));
    
    // Check if upload function was called
    expect(uploadBytesResumable).toHaveBeenCalledWith(
      expect.anything(),
      file,
      { contentType: 'video/mp4' }
    );
    
    // Check if progress was updated
    await waitFor(() => {
      expect(screen.getByText('Uploading: 50%')).toBeInTheDocument();
    });
  });
});
