import React from 'react';
import { fireEvent, waitFor, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '../../../test-utils';
import LandingPage from '../../../components/pages/LandingPage';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { getStorage, uploadBytesResumable } from 'firebase/storage';

// Mock useNavigate
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }) => <div>{children}</div>,
  Navigate: ({ to, replace }) => <div data-testid="navigate" data-to={to} data-replace={replace}>Redirecting...</div>,
  useNavigate: () => mockNavigate,
  useLocation: () => ({ state: null }),
  Link: ({ to, children }) => <a href={to}>{children}</a>
}));

describe('LandingPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders login form by default', () => {
    render(<LandingPage />);
    
    expect(screen.getByText('Welcome to Video Vault')).toBeInTheDocument();
    expect(screen.getByText('Sign In')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Email')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Password')).toBeInTheDocument();
    expect(screen.getByText('Continue with Google')).toBeInTheDocument();
  });

  test('toggles between sign in and sign up modes', async () => {
    render(<LandingPage />);
    
    // Initially in sign in mode
    expect(screen.getByText('Sign In')).toBeInTheDocument();
    
    // Click to switch to sign up mode
    fireEvent.click(screen.getByText('Sign Up'));
    
    // Now in sign up mode
    expect(screen.getByText('Create an Account')).toBeInTheDocument();
    
    // Click to switch back to sign in mode
    fireEvent.click(screen.getByText('Sign In'));
    
    // Back to sign in mode
    expect(screen.queryByText('Create an Account')).not.toBeInTheDocument();
  });

  test('handles email/password sign in', async () => {
    signInWithEmailAndPassword.mockResolvedValueOnce({ user: { uid: 'test-uid' } });
    
    render(<LandingPage />);
    
    // Fill in the form
    await userEvent.type(screen.getByPlaceholderText('Email'), 'test@example.com');
    await userEvent.type(screen.getByPlaceholderText('Password'), 'password123');
    
    // Submit the form
    fireEvent.click(screen.getByText('Sign In'));
    
    // Check if the sign in function was called with correct params
    await waitFor(() => {
      expect(signInWithEmailAndPassword).toHaveBeenCalledWith(
        expect.anything(),
        'test@example.com',
        'password123'
      );
    });
    
    // Check if navigation happened
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/second-page');
    });
  });

  test('handles email/password sign up', async () => {
    const mockUserCredential = { 
      user: { uid: 'test-uid' }
    };
    
    createUserWithEmailAndPassword.mockResolvedValueOnce(mockUserCredential);
    uploadBytesResumable.mockResolvedValueOnce({});
    
    render(<LandingPage />);
    
    // Switch to sign up mode
    fireEvent.click(screen.getByText('Sign Up'));
    
    // Fill in the form
    await userEvent.type(screen.getByPlaceholderText('Email'), 'newuser@example.com');
    await userEvent.type(screen.getByPlaceholderText('Password'), 'newpassword123');
    
    // Submit the form
    fireEvent.click(screen.getByText('Sign Up'));
    
    // Check if the sign up function was called with correct params
    await waitFor(() => {
      expect(createUserWithEmailAndPassword).toHaveBeenCalledWith(
        expect.anything(),
        'newuser@example.com',
        'newpassword123'
      );
    });
    
    // Check if navigation happened
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/second-page');
    });
  });

  test('handles Google sign in', async () => {
    const mockResult = {
      user: { uid: 'google-uid' },
      _tokenResponse: { isNewUser: false }
    };
    
    signInWithPopup.mockResolvedValueOnce(mockResult);
    
    render(<LandingPage />);
    
    // Click Google sign in button
    fireEvent.click(screen.getByText('Continue with Google'));
    
    // Check if the Google sign in function was called
    await waitFor(() => {
      expect(signInWithPopup).toHaveBeenCalled();
    });
    
    // Check if navigation happened
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/second-page');
    });
  });

  test('handles Google sign up (new user)', async () => {
    const mockResult = {
      user: { uid: 'google-uid' },
      _tokenResponse: { isNewUser: true }
    };
    
    signInWithPopup.mockResolvedValueOnce(mockResult);
    uploadBytesResumable.mockResolvedValueOnce({});
    
    render(<LandingPage />);
    
    // Click Google sign in button
    fireEvent.click(screen.getByText('Continue with Google'));
    
    // Check if the Google sign in function was called
    await waitFor(() => {
      expect(signInWithPopup).toHaveBeenCalled();
    });
    
    // Check if user bucket creation was attempted
    await waitFor(() => {
      expect(getStorage).toHaveBeenCalled();
    });
    
    // Check if navigation happened
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/second-page');
    });
  });
});
