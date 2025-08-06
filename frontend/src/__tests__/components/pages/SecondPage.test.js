import React from 'react';
import { fireEvent, waitFor, screen } from '@testing-library/react';
import { render } from '../../../test-utils';
import UploadPage from '../../../components/pages/SecondPage';
import { getAuth } from 'firebase/auth';
import { getStorage, ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';

// Mock useNavigate and useLocation
const mockNavigate = jest.fn();
const mockLocation = {
  state: { projectName: 'test-project' }
};

// Override the mock from test-utils.js
jest.spyOn(require('react-router-dom'), 'useNavigate').mockImplementation(() => mockNavigate);
jest.spyOn(require('react-router-dom'), 'useLocation').mockImplementation(() => mockLocation);

// Mock UUID
jest.mock('uuid', () => ({
  v4: jest.fn(() => 'mock-uuid')
}));

describe('UploadPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock the auth.currentUser
    getAuth.mockReturnValue({
      currentUser: { uid: 'test-user-id' }
    });
    
    // Mock URL.createObjectURL and URL.revokeObjectURL
    global.URL.createObjectURL = jest.fn(() => 'mock-url');
    global.URL.revokeObjectURL = jest.fn();
  });

  test('renders with project name from location state', () => {
    render(<UploadPage />);
    
    expect(screen.getByText('Upload to test-project')).toBeInTheDocument();
    expect(screen.getByLabelText('Project Name:')).toHaveValue('test-project');
  });

  test('renders with empty project name when not provided in location state', () => {
    // Override the mock location to simulate no project name in state
    mockLocation.state = null;
    
    render(<UploadPage />);
    
    expect(screen.getByText('Create New Project')).toBeInTheDocument();
    expect(screen.getByLabelText('Project Name:')).toHaveValue('');
  });

  test('allows user to enter project name', () => {
    mockLocation.state = null;
    
    render(<UploadPage />);
    
    const projectNameInput = screen.getByLabelText('Project Name:');
    fireEvent.change(projectNameInput, { target: { value: 'new-project' } });
    
    expect(projectNameInput).toHaveValue('new-project');
  });

  test('shows video preview when file is selected', async () => {
    render(<UploadPage />);
    
    const file = new File(['dummy content'], 'test-video.mp4', { type: 'video/mp4' });
    const fileInput = screen.getByLabelText('Choose Video File');
    
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    
    fireEvent.change(fileInput);
    
    await waitFor(() => {
      expect(screen.getByText('Video Preview')).toBeInTheDocument();
      expect(screen.getByText('test-video.mp4')).toBeInTheDocument();
      expect(screen.getByText('Upload Video')).toBeInTheDocument();
    });
  });

  test('disables upload button when project name is empty', async () => {
    mockLocation.state = null;
    
    render(<UploadPage />);
    
    const file = new File(['dummy content'], 'test-video.mp4', { type: 'video/mp4' });
    const fileInput = screen.getByLabelText('Choose Video File');
    
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    
    fireEvent.change(fileInput);
    
    await waitFor(() => {
      expect(screen.getByText('Enter Project Name First')).toBeInTheDocument();
      expect(screen.getByText('Enter Project Name First')).toBeDisabled();
    });
    
    // Enter project name
    const projectNameInput = screen.getByLabelText('Project Name:');
    fireEvent.change(projectNameInput, { target: { value: 'new-project' } });
    
    await waitFor(() => {
      expect(screen.getByText('Upload Video')).toBeInTheDocument();
      expect(screen.getByText('Upload Video')).not.toBeDisabled();
    });
  });

  test('uploads video when upload button is clicked', async () => {
    // Mock the upload functions
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
    
    render(<UploadPage />);
    
    const file = new File(['dummy content'], 'test-video.mp4', { type: 'video/mp4' });
    const fileInput = screen.getByLabelText('Choose Video File');
    
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    
    fireEvent.change(fileInput);
    
    await waitFor(() => {
      expect(screen.getByText('Upload Video')).toBeInTheDocument();
    });
    
    // Click upload button
    fireEvent.click(screen.getByText('Upload Video'));
    
    // Check if upload function was called with correct params
    expect(uploadBytesResumable).toHaveBeenCalledWith(
      expect.anything(),
      file,
      { contentType: 'video/mp4' }
    );
    
    // Check if progress was updated
    await waitFor(() => {
      expect(screen.getByText('Uploading: 50%')).toBeInTheDocument();
    });
    
    // Check if navigation happened after upload complete
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/video-page', { state: { projectName: 'test-project' } });
    });
  });

  test('navigates back when cancel button is clicked', () => {
    render(<UploadPage />);
    
    // Click cancel button
    fireEvent.click(screen.getByText('Cancel'));
    
    // Should navigate back to video page with project name
    expect(mockNavigate).toHaveBeenCalledWith('/video-page', { state: { projectName: 'test-project' } });
  });

  test('navigates to projects page when cancel button is clicked with no project', () => {
    mockLocation.state = null;
    
    render(<UploadPage />);
    
    // Click cancel button
    fireEvent.click(screen.getByText('Cancel'));
    
    // Should navigate back to projects page
    expect(mockNavigate).toHaveBeenCalledWith('/projects');
  });
});
