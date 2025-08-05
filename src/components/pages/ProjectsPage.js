import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAuth } from 'firebase/auth';
import { getStorage, ref, listAll } from 'firebase/storage';
import { toast } from 'react-toastify';
import './VideoPage.css';
import './SecondPage.css'; // For common styles

function ProjectsPage() {
  const auth = getAuth();
  const user = auth.currentUser;
  const navigate = useNavigate();
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch user's projects
  useEffect(() => {
    if (!user) return;
    
    const fetchProjects = async () => {
      try {
        setLoading(true);
        const storage = getStorage();
        const userStorageRef = ref(storage, user.uid);
        
        // List all projects (folders) in the user's storage
        const projectsResult = await listAll(userStorageRef);
        
        // For each project, get its videos count
        const projectsData = await Promise.all(
          projectsResult.prefixes.map(async (projectRef) => {
            const projectName = projectRef.name;
            const videosResult = await listAll(projectRef);
            
            return {
              name: projectName,
              videoCount: videosResult.items.length
            };
          })
        );
        
        setProjects(projectsData);
      } catch (error) {
        console.error('Error fetching projects:', error);
        toast.error('Failed to load projects. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchProjects();
  }, [user]);
  
  const handleProjectSelect = (projectName) => {
    navigate('/video-page', { state: { projectName } });
  };

  const handleCreateProject = () => {
    navigate('/upload-project');
  };

  if (!user) {
    return null; // or a loading spinner
  }

  return (
    <div className="second-page">
      <div className="user-card">
        <h1>Your Projects</h1>
        
        <div className="projects-container">
          {loading ? (
            <div className="loading-spinner"></div>
          ) : projects.length === 0 ? (
            <div className="empty-projects">
              <p>You don't have any projects yet.</p>
              <button onClick={handleCreateProject} className="create-project-button">
                Create Your First Project
              </button>
            </div>
          ) : (
            <>
              <div className="projects-header">
                <h2>All Projects</h2>
                <button onClick={handleCreateProject} className="create-project-button">
                  + New Project
                </button>
              </div>
              <div className="projects-list">
                {projects.map((project) => (
                  <div 
                    key={project.name} 
                    className="project-item" 
                    onClick={() => handleProjectSelect(project.name)}
                  >
                    <h3>{project.name}</h3>
                    <p>{project.videoCount} video{project.videoCount !== 1 ? 's' : ''}</p>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
        
        <div className="button-group">
          <button onClick={() => navigate('/')} className="back-button">
            Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}

export default ProjectsPage;
