import React from 'react';
import { fireEvent, waitFor, screen } from '@testing-library/react';
import { render } from '../../../test-utils';
import ProjectsPage from '../../../components/pages/ProjectsPage';
import { getAuth } from 'firebase/auth';
import { getStorage, ref, listAll } from 'firebase/storage';

// Mock useNavigate
const mockNavigate = jest.fn();
// Mock is already defined in test-utils.js

describe('ProjectsPage Component', () => {
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
    
    render(<ProjectsPage />);
    
    expect(screen.getByText('Your Projects')).toBeInTheDocument();
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  test('renders empty state when no projects exist', async () => {
    // Mock empty projects list
    listAll.mockResolvedValue({ prefixes: [] });
    
    render(<ProjectsPage />);
    
    await waitFor(() => {
      expect(screen.getByText("You don't have any projects yet.")).toBeInTheDocument();
      expect(screen.getByText("Create Your First Project")).toBeInTheDocument();
    });
  });

  test('renders projects list when projects exist', async () => {
    // Mock projects list
    const mockPrefixes = [
      { name: 'project1' },
      { name: 'project2' }
    ];
    
    listAll.mockImplementation((ref) => {
      if (ref.fullPath === 'test-user-id') {
        return Promise.resolve({ prefixes: mockPrefixes });
      } else if (ref.fullPath === 'test-user-id/project1') {
        return Promise.resolve({ items: [{ name: 'video1.mp4' }, { name: 'video2.mp4' }] });
      } else if (ref.fullPath === 'test-user-id/project2') {
        return Promise.resolve({ items: [{ name: 'video3.mp4' }] });
      }
      return Promise.resolve({ items: [] });
    });
    
    ref.mockImplementation((storage, path) => {
      return { fullPath: path, name: path.split('/').pop() };
    });
    
    render(<ProjectsPage />);
    
    await waitFor(() => {
      expect(screen.getByText('All Projects')).toBeInTheDocument();
      expect(screen.getByText('project1')).toBeInTheDocument();
      expect(screen.getByText('project2')).toBeInTheDocument();
      expect(screen.getByText('2 videos')).toBeInTheDocument();
      expect(screen.getByText('1 video')).toBeInTheDocument();
    });
  });

  test('navigates to video page when project is selected', async () => {
    // Mock projects list
    const mockPrefixes = [
      { name: 'project1' }
    ];
    
    listAll.mockImplementation((ref) => {
      if (ref.fullPath === 'test-user-id') {
        return Promise.resolve({ prefixes: mockPrefixes });
      } else if (ref.fullPath === 'test-user-id/project1') {
        return Promise.resolve({ items: [{ name: 'video1.mp4' }] });
      }
      return Promise.resolve({ items: [] });
    });
    
    ref.mockImplementation((storage, path) => {
      return { fullPath: path, name: path.split('/').pop() };
    });
    
    render(<ProjectsPage />);
    
    await waitFor(() => {
      expect(screen.getByText('project1')).toBeInTheDocument();
    });
    
    // Click on the project
    fireEvent.click(screen.getByText('project1'));
    
    expect(mockNavigate).toHaveBeenCalledWith('/video-page', { state: { projectName: 'project1' } });
  });

  test('navigates to upload page when create project button is clicked', async () => {
    // Mock empty projects list
    listAll.mockResolvedValue({ prefixes: [] });
    
    render(<ProjectsPage />);
    
    await waitFor(() => {
      expect(screen.getByText("Create Your First Project")).toBeInTheDocument();
    });
    
    // Click create project button
    fireEvent.click(screen.getByText("Create Your First Project"));
    
    expect(mockNavigate).toHaveBeenCalledWith('/upload-project');
  });

  test('navigates to home when back button is clicked', async () => {
    // Mock projects list
    listAll.mockResolvedValue({ prefixes: [] });
    
    render(<ProjectsPage />);
    
    // Click back button
    fireEvent.click(screen.getByText('Back to Home'));
    
    expect(mockNavigate).toHaveBeenCalledWith('/');
  });
});
