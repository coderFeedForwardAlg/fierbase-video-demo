import React from 'react';
import { render, screen } from '@testing-library/react';
import ProtectedRoute from '../../../components/common/ProtectedRoute';
import { getAuth } from 'firebase/auth';
import { Navigate } from 'react-router-dom';

// Mock Navigate component
jest.mock('react-router-dom', () => ({
  Navigate: jest.fn(() => <div data-testid="navigate">Redirecting...</div>)
}));

describe('ProtectedRoute Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('redirects to home when user is not authenticated', () => {
    // Mock auth with no user
    getAuth.mockReturnValue({
      currentUser: null
    });

    render(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected Content</div>
      </ProtectedRoute>
    );

    // Should redirect to home
    expect(Navigate).toHaveBeenCalledWith({ to: '/', replace: true }, {});
    expect(screen.getByTestId('navigate')).toBeInTheDocument();
    expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
  });

  test('renders children when user is authenticated', () => {
    // Mock auth with authenticated user
    getAuth.mockReturnValue({
      currentUser: { uid: 'test-user' }
    });

    render(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected Content</div>
      </ProtectedRoute>
    );

    // Should render children
    expect(Navigate).not.toHaveBeenCalled();
    expect(screen.queryByTestId('navigate')).not.toBeInTheDocument();
    expect(screen.getByTestId('protected-content')).toBeInTheDocument();
  });
});
