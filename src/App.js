import './App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import { useState, useEffect } from 'react';
import LandingPage from './components/pages/LandingPage';
import ProjectsPage from './components/pages/ProjectsPage';
import VideoPage from './components/pages/VideoPage';
import UploadPage from './components/pages/SecondPage'; // Using the existing file with new component name
import ProtectedRoute from './components/common/ProtectedRoute';
import { initializeApp } from "firebase/app";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDmyZJx4hESeYseCGpyG4hI8DQo3z2zPRA",
  authDomain: "video-vault-gnytb.firebaseapp.com",
  projectId: "video-vault-gnytb",
  storageBucket: "video-vault-gnytb.firebasestorage.app",
  messagingSenderId: "1052283877831",
  appId: "1:1052283877831:web:18f4959acaeb48fb86c1e2"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

function App() {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const auth = getAuth();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setIsLoading(false);
    });

    return () => unsubscribe();
  }, [auth]);

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route 
          path="/" 
          element={user ? <Navigate to="/projects" replace /> : <LandingPage />} 
        />
        <Route 
          path="/projects" 
          element={
            <ProtectedRoute>
              <ProjectsPage />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/video-page" 
          element={
            <ProtectedRoute>
              <VideoPage />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/upload-project" 
          element={
            <ProtectedRoute>
              <UploadPage />
            </ProtectedRoute>
          } 
        />
        {/* Keep the old route for backward compatibility */}
        <Route 
          path="/second-page" 
          element={
            <ProtectedRoute>
              <Navigate to="/projects" replace />
            </ProtectedRoute>
          } 
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
