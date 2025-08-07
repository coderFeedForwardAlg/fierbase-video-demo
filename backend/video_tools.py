"""
Video manipulation tools using ffmpeg for Genkit tool calling
"""
import os
import tempfile
import logging
import uuid
import ffmpeg
import firebase_admin
from firebase_admin import storage
import urllib.request
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoTools:
    """
    Class containing video manipulation tools that can be called by Genkit
    """
    
    def __init__(self, storage_bucket: str = None):
        """
        Initialize video tools with Firebase Storage bucket
        
        Args:
            storage_bucket: Firebase Storage bucket name
        """
        self.temp_dir = tempfile.mkdtemp()
        self.storage_bucket = storage_bucket
        
    def _download_video(self, video_url: str) -> str:
        """
        Download video from URL to temporary file
        
        Args:
            video_url: URL of the video to download
            
        Returns:
            Path to downloaded video file
        """
        # Create a temporary file for the video
        temp_video_path = os.path.join(self.temp_dir, f"input_{uuid.uuid4()}.mp4")
        
        # Download the video
        try:
            urllib.request.urlretrieve(video_url, temp_video_path)
            logger.info(f"Downloaded video to {temp_video_path}")
            return temp_video_path
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
    
    def _upload_to_firebase(self, file_path: str, destination_path: str) -> str:
        """
        Upload a file to Firebase Storage
        
        Args:
            file_path: Local path to the file
            destination_path: Path in Firebase Storage
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            bucket = storage.bucket(name=self.storage_bucket)
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(file_path)
            
            # Make the file publicly accessible
            blob.make_public()
            
            # Get the public URL
            public_url = blob.public_url
            logger.info(f"Uploaded file to {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"Error uploading to Firebase: {str(e)}")
            raise
    
    def trim_video(self, video_url: str, start_time: float, end_time: float, 
                  user_id: str, project_name: str, output_name: str = None) -> Dict[str, Any]:
        """
        Trim a video to the specified start and end times
        
        Args:
            video_url: URL of the video to trim
            start_time: Start time in seconds
            end_time: End time in seconds
            user_id: Firebase user ID
            project_name: Project name
            output_name: Name for the output file (optional)
            
        Returns:
            Dictionary with information about the trimmed video
        """
        try:
            # Download the video
            input_path = self._download_video(video_url)
            
            # Generate output filename
            if not output_name:
                output_name = f"trimmed_{os.path.basename(input_path)}"
            output_path = os.path.join(self.temp_dir, output_name)
            
            # Trim the video using ffmpeg
            (
                ffmpeg
                .input(input_path)
                .output(output_path, ss=start_time, to=end_time, c='copy')
                .run(quiet=True, overwrite_output=True)
            )
            
            # Upload to Firebase Storage
            destination_path = f"{user_id}/{project_name}/{output_name}"
            public_url = self._upload_to_firebase(output_path, destination_path)
            
            # Clean up temporary files
            os.remove(input_path)
            os.remove(output_path)
            
            return {
                "success": True,
                "operation": "trim_video",
                "output_name": output_name,
                "output_url": public_url,
                "start_time": start_time,
                "end_time": end_time
            }
        except Exception as e:
            logger.error(f"Error trimming video: {str(e)}")
            return {
                "success": False,
                "operation": "trim_video",
                "error": str(e)
            }
    
    def change_speed(self, video_url: str, speed_factor: float,
                    user_id: str, project_name: str, output_name: str = None) -> Dict[str, Any]:
        """
        Change the speed of a video
        
        Args:
            video_url: URL of the video
            speed_factor: Speed factor (0.5 = half speed, 2.0 = double speed)
            user_id: Firebase user ID
            project_name: Project name
            output_name: Name for the output file (optional)
            
        Returns:
            Dictionary with information about the modified video
        """
        try:
            # Download the video
            input_path = self._download_video(video_url)
            
            # Generate output filename
            if not output_name:
                output_name = f"speed_{speed_factor}x_{os.path.basename(input_path)}"
            output_path = os.path.join(self.temp_dir, output_name)
            
            # Change video speed using ffmpeg
            # For speed_factor > 1, we're speeding up (setpts=1/speed_factor)
            # For speed_factor < 1, we're slowing down (setpts=1/speed_factor)
            (
                ffmpeg
                .input(input_path)
                .output(output_path, 
                        vf=f"setpts={1/speed_factor}*PTS", 
                        af=f"atempo={speed_factor}")
                .run(quiet=True, overwrite_output=True)
            )
            
            # Upload to Firebase Storage
            destination_path = f"{user_id}/{project_name}/{output_name}"
            public_url = self._upload_to_firebase(output_path, destination_path)
            
            # Clean up temporary files
            os.remove(input_path)
            os.remove(output_path)
            
            return {
                "success": True,
                "operation": "change_speed",
                "output_name": output_name,
                "output_url": public_url,
                "speed_factor": speed_factor
            }
        except Exception as e:
            logger.error(f"Error changing video speed: {str(e)}")
            return {
                "success": False,
                "operation": "change_speed",
                "error": str(e)
            }
    
    def add_text_overlay(self, video_url: str, text: str, user_id: str, project_name: str, position: str = "center", output_name: str = None) -> Dict[str, Any]:
        """
        Add text overlay to a video
        
        Args:
            video_url: URL of the video
            text: Text to overlay
            position: Position of the text (top, bottom, center)
            user_id: Firebase user ID
            project_name: Project name
            output_name: Name for the output file (optional)
            
        Returns:
            Dictionary with information about the modified video
        """
        try:
            # Download the video
            input_path = self._download_video(video_url)
            
            # Generate output filename
            if not output_name:
                output_name = f"text_{os.path.basename(input_path)}"
            output_path = os.path.join(self.temp_dir, output_name)
            
            # Set position coordinates
            position_map = {
                "top": "x=(w-text_w)/2:y=20",
                "bottom": "x=(w-text_w)/2:y=h-th-20",
                "center": "x=(w-text_w)/2:y=(h-text_h)/2"
            }
            pos = position_map.get(position.lower(), position_map["bottom"])
            
            # Add text overlay using ffmpeg
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vf=f"drawtext=text='{text}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:{pos}")
                .run(quiet=True, overwrite_output=True)
            )
            
            # Upload to Firebase Storage
            destination_path = f"{user_id}/{project_name}/{output_name}"
            public_url = self._upload_to_firebase(output_path, destination_path)
            
            # Clean up temporary files
            os.remove(input_path)
            os.remove(output_path)
            
            return {
                "success": True,
                "operation": "add_text_overlay",
                "output_name": output_name,
                "output_url": public_url,
                "text": text,
                "position": position
            }
        except Exception as e:
            logger.error(f"Error adding text overlay: {str(e)}")
            return {
                "success": False,
                "operation": "add_text_overlay",
                "error": str(e)
            }
    
    def rotate_video(self, video_url: str, degrees: int,
                    user_id: str, project_name: str, output_name: str = None) -> Dict[str, Any]:
        """
        Rotate a video by the specified degrees
        
        Args:
            video_url: URL of the video
            degrees: Rotation angle in degrees (90, 180, 270)
            user_id: Firebase user ID
            project_name: Project name
            output_name: Name for the output file (optional)
            
        Returns:
            Dictionary with information about the rotated video
        """
        try:
            # Download the video
            input_path = self._download_video(video_url)
            
            # Generate output filename
            if not output_name:
                output_name = f"rotated_{degrees}_{os.path.basename(input_path)}"
            output_path = os.path.join(self.temp_dir, output_name)
            
            # Map degrees to transpose values
            transpose_map = {
                90: "1",    # 90 degrees clockwise
                180: "2,2", # 180 degrees (apply 90 degrees twice)
                270: "2"    # 90 degrees counterclockwise
            }
            
            transpose = transpose_map.get(degrees)
            if not transpose:
                raise ValueError(f"Unsupported rotation angle: {degrees}. Use 90, 180, or 270.")
            
            # Rotate the video using ffmpeg
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vf=f"transpose={transpose}")
                .run(quiet=True, overwrite_output=True)
            )
            
            # Upload to Firebase Storage
            destination_path = f"{user_id}/{project_name}/{output_name}"
            public_url = self._upload_to_firebase(output_path, destination_path)
            
            # Clean up temporary files
            os.remove(input_path)
            os.remove(output_path)
            
            return {
                "success": True,
                "operation": "rotate_video",
                "output_name": output_name,
                "output_url": public_url,
                "degrees": degrees
            }
        except Exception as e:
            logger.error(f"Error rotating video: {str(e)}")
            return {
                "success": False,
                "operation": "rotate_video",
                "error": str(e)
            }
