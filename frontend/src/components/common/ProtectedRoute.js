import { Navigate } from 'react-router-dom';
import { getAuth } from 'firebase/auth';

export default function ProtectedRoute({ children }) {
  const auth = getAuth();
  const user = auth.currentUser;

  if (!user) {
    // Redirect to the login page if user is not authenticated
    return <Navigate to="/" replace />;
  }

  return children;
}
