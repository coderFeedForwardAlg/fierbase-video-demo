import React from 'react';
import { render } from '@testing-library/react';

// Mock react-router-dom
jest.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }) => <div>{children}</div>,
  Navigate: ({ to, replace }) => <div data-testid="navigate" data-to={to} data-replace={replace}>Redirecting...</div>,
  useNavigate: () => jest.fn(),
  useLocation: () => ({ state: null }),
  Link: ({ to, children }) => <a href={to}>{children}</a>
}));

// Mock Firebase modules
jest.mock('firebase/app', () => {
  return {
    initializeApp: jest.fn().mockReturnValue({
      name: 'test-app',
      options: {}
    })
  };
});

jest.mock('firebase/auth', () => {
  return {
    getAuth: jest.fn().mockReturnValue({
      currentUser: null,
      onAuthStateChanged: jest.fn()
    }),
    onAuthStateChanged: jest.fn(),
    signInWithEmailAndPassword: jest.fn(),
    createUserWithEmailAndPassword: jest.fn(),
    signInWithPopup: jest.fn(),
    GoogleAuthProvider: jest.fn().mockImplementation(() => {
      return {};
    }),
    signOut: jest.fn()
  };
});

jest.mock('firebase/storage', () => {
  return {
    getStorage: jest.fn().mockReturnValue({}),
    ref: jest.fn(),
    uploadBytesResumable: jest.fn(),
    getDownloadURL: jest.fn(),
    listAll: jest.fn()
  };
});

// Mock react-toastify
jest.mock('react-toastify', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    info: jest.fn()
  },
  ToastContainer: () => <div data-testid="toast-container" />
}));

// Custom render
const customRender = (ui, options) => {
  return render(ui, options);
};

// Re-export everything
export * from '@testing-library/react';

// Override render method
export { customRender as render };
