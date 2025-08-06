import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import App from '../App';
import { getAuth } from 'firebase/auth';

// Mock BrowserRouter and Routes
jest.mock('react-router-dom', () => {
  const originalModule = jest.requireActual('react-router-dom');
  return {
    ...originalModule,
    BrowserRouter: ({ children }) => <div>{children}</div>,
  };
});

// Mock the components
jest.mock('../components/pages/LandingPage', () => () => <div data-testid="landing-page">Landing Page</div>);
jest.mock('../components/pages/ProjectsPage', () => () => <div data-testid="projects-page">Projects Page</div>);
jest.mock('../components/pages/VideoPage', () => () => <div data-testid="video-page">Video Page</div>);
jest.mock('../components/pages/SecondPage', () => () => <div data-testid="upload-page">Upload Page</div>);

describe('App Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    // Mock auth with loading state
    getAuth.mockReturnValue({
      currentUser: null,
      onAuthStateChanged: jest.fn(() => jest.fn()) // Never calls the callback
    });

    render(<App />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  test('redirects to landing page when user is not authenticated', async () => {
    // Mock auth with no user
    getAuth.mockReturnValue({
      currentUser: null,
      onAuthStateChanged: jest.fn((auth, callback) => {
        callback(null);
        return jest.fn();
      })
    });

    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByTestId('landing-page')).toBeInTheDocument();
    });
  });

  test('redirects to projects page when user is authenticated', async () => {
    // Mock auth with authenticated user
    getAuth.mockReturnValue({
      currentUser: { uid: 'test-user' },
      onAuthStateChanged: jest.fn((auth, callback) => {
        callback({ uid: 'test-user' });
        return jest.fn();
      })
    });

    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByTestId('projects-page')).toBeInTheDocument();
    });
  });
});
