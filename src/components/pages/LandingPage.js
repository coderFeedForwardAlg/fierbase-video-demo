import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import { getStorage, ref, getDownloadURL, uploadBytesResumable } from 'firebase/storage';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './LandingPage.css';

function LandingPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const auth = getAuth();
  const navigate = useNavigate();
  const googleProvider = new GoogleAuthProvider();

  const createUserBucket = async (userId) => {
    try {
      const storage = getStorage();
      // Create a reference to the user's root folder
      const userRootRef = ref(storage, `${userId}/`);
      // Create a default project folder
      const defaultProjectRef = ref(storage, `${userId}/default_project/`);
      
      // Create a placeholder file to ensure the folder exists
      const placeholderFile = new File([''], '.keep', { type: 'text/plain' });
      const placeholderRef = ref(defaultProjectRef, '.keep');
      
      // Upload the placeholder file
      await uploadBytesResumable(placeholderRef, placeholderFile);
      console.log('User bucket and default project created successfully');
    } catch (error) {
      console.error('Error creating user bucket:', error);
      throw error; // Re-throw to be caught by the calling function
    }
  };

  const handleEmailPasswordAuth = async (e) => {
    e.preventDefault();
    try {
      if (isSignUp) {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        // Create user bucket after successful account creation
        await createUserBucket(userCredential.user.uid);
        toast.success('Account created successfully!');
      } else {
        await signInWithEmailAndPassword(auth, email, password);
        toast.success('Signed in successfully!');
      }
      navigate('/second-page');
    } catch (error) {
      toast.error(error.message);
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      // Check if this is a new user
      if (result._tokenResponse?.isNewUser) {
        // Create user bucket for new Google sign-up
        await createUserBucket(result.user.uid);
      }
      toast.success('Signed in with Google!');
      navigate('/second-page');
    } catch (error) {
      toast.error(error.message);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h1>Welcome to Video Vault</h1>
        <h2>{isSignUp ? 'Create an Account' : 'Sign In'}</h2>
        
        <form onSubmit={handleEmailPasswordAuth} className="auth-form">
          <div className="form-group">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              required
            />
          </div>
          <div className="form-group">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              required
              minLength="6"
            />
          </div>
          <button type="submit" className="auth-button">
            {isSignUp ? 'Sign Up' : 'Sign In'}
          </button>
        </form>

        <button onClick={handleGoogleSignIn} className="google-button">
          <img src="https://www.google.com/favicon.ico" alt="Google" className="google-icon" />
          Continue with Google
        </button>

        <p className="auth-toggle">
          {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
          <span onClick={() => setIsSignUp(!isSignUp)} className="toggle-link">
            {isSignUp ? 'Sign In' : 'Sign Up'}
          </span>
        </p>
      </div>
      <ToastContainer position="top-right" autoClose={3000} />
    </div>
  );
}

export default LandingPage;
