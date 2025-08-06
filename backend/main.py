from fastapi import FastAPI, HTTPException, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import os
import json
import logging
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage

# Google AI
import google.generativeai as genai

# Video tools
from video_tools import (
    trim_video, 
    change_speed, 
    add_text_overlay, 
    rotate_video
)

# Load environment variables
load_dotenv(".env/.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
try:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not os.path.exists(cred_path):
        logger.error(f"Firebase credentials file not found at {cred_path}")
        raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")
    
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
    })
    logger.info("Firebase Admin SDK initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Firebase Admin SDK: {str(e)}")
    # Continue without Firebase for development purposes
    pass

# Initialize video tools
try:
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    video_tools = VideoTools(storage_bucket=bucket_name)
    logger.info("Video tools initialized successfully")
except Exception as e:
    logger.error(f"Error initializing video tools: {str(e)}")
    video_tools = None

# Define tool functions for Genkit
def trim_video_tool(video_url: str, start_time: float, end_time: float, 
                  user_id: str, project_name: str, output_name: Optional[str] = None) -> Dict[str, Any]:
    """Trim a video to the specified start and end times
    
    Args:
        video_url: URL of the video to trim
        start_time: Start time in seconds
        end_time: End time in seconds
        user_id: Firebase user ID
        project_name: Project name
        output_name: Name for the output file (optional)
    """
    if video_tools is None:
        return {"success": False, "error": "Video tools not initialized"}
    
    return video_tools.trim_video(
        video_url=video_url,
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        project_name=project_name,
        output_name=output_name
    )

def change_speed_tool(video_url: str, speed_factor: float,
                    user_id: str, project_name: str, output_name: Optional[str] = None) -> Dict[str, Any]:
    """Change the speed of a video
    
    Args:
        video_url: URL of the video
        speed_factor: Speed factor (0.5 = half speed, 2.0 = double speed)
        user_id: Firebase user ID
        project_name: Project name
        output_name: Name for the output file (optional)
    """
    if video_tools is None:
        return {"success": False, "error": "Video tools not initialized"}
    
    return video_tools.change_speed(
        video_url=video_url,
        speed_factor=speed_factor,
        user_id=user_id,
        project_name=project_name,
        output_name=output_name
    )

def add_text_overlay_tool(video_url: str, text: str, user_id: str, project_name: str, 
                        position: str = "bottom", output_name: Optional[str] = None) -> Dict[str, Any]:
    """Add text overlay to a video
    
    Args:
        video_url: URL of the video
        text: Text to overlay
        position: Position of the text (top, bottom, center)
        user_id: Firebase user ID
        project_name: Project name
        output_name: Name for the output file (optional)
    """
    if video_tools is None:
        return {"success": False, "error": "Video tools not initialized"}
    
    return video_tools.add_text_overlay(
        video_url=video_url,
        text=text,
        position=position,
        user_id=user_id,
        project_name=project_name,
        output_name=output_name
    )

def rotate_video_tool(video_url: str, degrees: int,
                    user_id: str, project_name: str, output_name: Optional[str] = None) -> Dict[str, Any]:
    """Rotate a video by the specified degrees
    
    Args:
        video_url: URL of the video
        degrees: Rotation angle in degrees (90, 180, 270)
        user_id: Firebase user ID
        project_name: Project name
        output_name: Name for the output file (optional)
    """
    if video_tools is None:
        return {"success": False, "error": "Video tools not initialized"}
    
    return video_tools.rotate_video(
        video_url=video_url,
        degrees=degrees,
        user_id=user_id,
        project_name=project_name,
        output_name=output_name
    )

# Initialize Google Generative AI and register tools
try:
    # Define the tools schema for Google Generative AI
    tools_schema = [
        {
            "name": "trim_video",
            "description": "Trim a video to specific start and end times",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "URL of the video to trim"},
                    "start_time": {"type": "number", "description": "Start time in seconds"},
                    "end_time": {"type": "number", "description": "End time in seconds"},
                    "user_id": {"type": "string", "description": "Firebase user ID"},
                    "project_name": {"type": "string", "description": "Project name"},
                    "output_name": {"type": "string", "description": "Name for the output file (optional)"}
                },
                "required": ["video_url", "start_time", "end_time", "user_id", "project_name"]
            }
        },
        {
            "name": "change_speed",
            "description": "Change the playback speed of a video",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "URL of the video"},
                    "speed_factor": {"type": "number", "description": "Speed factor (0.5 = half speed, 2.0 = double speed)"},
                    "user_id": {"type": "string", "description": "Firebase user ID"},
                    "project_name": {"type": "string", "description": "Project name"},
                    "output_name": {"type": "string", "description": "Name for the output file (optional)"}
                },
                "required": ["video_url", "speed_factor", "user_id", "project_name"]
            }
        },
        {
            "name": "add_text_overlay",
            "description": "Add text overlay to a video",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "URL of the video"},
                    "text": {"type": "string", "description": "Text to overlay"},
                    "position": {"type": "string", "description": "Position of the text (top, bottom, center)"},
                    "user_id": {"type": "string", "description": "Firebase user ID"},
                    "project_name": {"type": "string", "description": "Project name"},
                    "output_name": {"type": "string", "description": "Name for the output file (optional)"}
                },
                "required": ["video_url", "text", "user_id", "project_name"]
            }
        },
        {
            "name": "rotate_video",
            "description": "Rotate a video by 90, 180, or 270 degrees",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "URL of the video"},
                    "degrees": {"type": "integer", "description": "Rotation angle in degrees (90, 180, 270)"},
                    "user_id": {"type": "string", "description": "Firebase user ID"},
                    "project_name": {"type": "string", "description": "Project name"},
                    "output_name": {"type": "string", "description": "Name for the output file (optional)"}
                },
                "required": ["video_url", "degrees", "user_id", "project_name"]
            }
        }
    ]
    
    # Map tool names to functions
    tools_map = {
        "trim_video": trim_video_tool,
        "change_speed": change_speed_tool,
        "add_text_overlay": add_text_overlay_tool,
        "rotate_video": rotate_video_tool
    }
    
    # Initialize Google Generative AI
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("Google Generative AI initialized successfully with tools")
except Exception as e:
    logger.error(f"Error initializing Google Generative AI: {str(e)}")

app = FastAPI(title="Video Vault API")

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "https://video-vault-gnytb.web.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000"],  # Add localhost for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class VideoAnalysisRequest(BaseModel):
    userId: str
    projectName: str
    videoName: str

class VideoAnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    error: Optional[str] = None
    
class VideoToolRequest(BaseModel):
    userId: str
    projectName: str
    videoUrl: str
    toolInstructions: str
    
class VideoToolResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to Video Vault API"}

@app.get("/ping")
async def ping():
    return {"status": "success", "message": "pong"}

@app.post("/api/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze a video from Firebase Storage using Genkit and Gemini 1.5"""
    try:
        # Check if Google API key is configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Google API key not configured")
        
        # Construct the path to the video in Firebase Storage
        video_path = f"{request.userId}/{request.projectName}/{request.videoName}"
        logger.info(f"Analyzing video at path: {video_path}")
        
        # Get a reference to the video in Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(video_path)
        
        # Check if the video exists
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        
        # Generate a signed URL for the video (valid for 10 minutes)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=600,  # 10 minutes
            method="GET"
        )
        
        # Use Google Generative AI to analyze the video with Gemini 1.5
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = await model.generate_content_async(
            contents=[
                {"mime_type": "video/mp4", "uri": signed_url},
                {"text": "Please analyze this video and provide the following information:\n" +
                         "1. A brief description of what's happening in the video\n" +
                         "2. Key objects and people visible\n" +
                         "3. Any notable events or actions\n" +
                         "4. Suggested edits or improvements that could be made\n" +
                         "Format your response as a JSON object with these sections."}
            ],
            generation_config={
                "temperature": 0.2,  # Lower temperature for more factual responses
            }
        )
        
        # Extract the analysis from the response
        analysis_text = response.text if hasattr(response, 'text') else ''
        
        # If using the new Google Generative AI API
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            analysis_text = part.text
                            break
        
        # Try to parse the response as JSON
        try:
            analysis_json = json.loads(analysis_text)
        except json.JSONDecodeError:
            # If the response is not valid JSON, return it as plain text
            analysis_json = {"raw_analysis": analysis_text}
        
        return VideoAnalysisResponse(analysis=analysis_json)
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return VideoAnalysisResponse(analysis={}, error=str(e))

@app.post("/api/video-tools", response_model=VideoToolResponse)
async def use_video_tools(request: VideoToolRequest):
    """Use AI to analyze a video and apply suggested tools"""
    try:
        # Check if Google API key is configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Google API key not configured")
        
        # Check if tools are registered
        if not tools_map or not tools_schema:
            raise HTTPException(status_code=500, detail="Video tools not initialized")
        
        # Use Google Generative AI to analyze the video and suggest tools
        model = genai.GenerativeModel(
            'gemini-1.5-pro',
            tools=tools_schema
        )
        
        response = await model.generate_content_async(
            contents=[
                {"mime_type": "video/mp4", "uri": request.videoUrl},
                {"text": f"Analyze this video and {request.toolInstructions}\n\n" +
                         "You have access to the following video editing tools:\n" +
                         "1. trim_video - Trim a video to specific start and end times\n" +
                         "2. change_speed - Change the playback speed of a video\n" +
                         "3. add_text_overlay - Add text overlay to a video\n" +
                         "4. rotate_video - Rotate a video by 90, 180, or 270 degrees\n\n" +
                         "Based on your analysis and the user's instructions, call the appropriate tool(s) to edit the video."}
            ],
            generation_config={
                "temperature": 0.2,
            }
        )
        
        # Process tool calls
        results = []
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            tool_name = part.function_call.name
                            tool_args = json.loads(part.function_call.args)
                            
                            # Add user ID and project name if not provided
                            if 'user_id' not in tool_args:
                                tool_args['user_id'] = request.userId
                            if 'project_name' not in tool_args:
                                tool_args['project_name'] = request.projectName
                            
                            # Execute the tool
                            if tool_name in tools_map:
                                tool_function = tools_map[tool_name]
                                result = tool_function(**tool_args)
                                results.append({
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "result": result
                                })
                            else:
                                results.append({
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "result": {"success": False, "error": f"Unknown tool: {tool_name}"}
                                })
        
        return VideoToolResponse(
            success=True,
            results=results
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error using video tools: {str(e)}")
        return VideoToolResponse(
            success=False,
            results=[],
            error=str(e)
        )

# Health check endpoint for the Genkit service
@app.get("/api/genkit-status")
async def genkit_status():
    """Check if Google Generative AI is properly initialized"""
    try:
        # Check if Google API key is configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"status": "error", "message": "Google API key not configured"}
            
        # Create a test model to verify API access
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        return {"status": "success", "message": "Google Generative AI initialized and ready"}
    except Exception as e:
        logger.error(f"Error checking Google Generative AI status: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
