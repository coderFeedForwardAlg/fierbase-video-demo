import os
import base64
import json
import re
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class VideoEvent(BaseModel):
    event_description: str = Field(description="Description of the event")
    start_time: float = Field(description="Start time of the event in seconds")
    end_time: float = Field(description="End time of the event in seconds")


class VideoAnalysisResult(BaseModel):
    events: List[VideoEvent] = Field(description="List of events in the video with timestamps")


class GeminiVideoAnalyzer:
    """Analyzer that uses Gemini via langchain-google-genai to extract time-stamped events from video bytes."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.parser = JsonOutputParser(pydantic_object=VideoAnalysisResult)

    def analyze_video_bytes(self, data: bytes, mime_type: str = "video/mp4") -> Dict[str, Any]:
        """Analyze raw video bytes and return a dict with an `events` list.

        Returns a structure compatible with VideoAnalysisResult.dict().
        """
        encoded_video = base64.b64encode(data).decode("utf-8")

        prompt = (
            "Analyze the provided video and identify distinct events that occur.\n"
            "For each event, provide: (1) a brief description, (2) start time in seconds, (3) end time in seconds.\n"
            "Be precise and only include meaningful events.\n\n"
            "Return ONLY JSON in this exact schema:\n"
            "{\n  \"events\": [\n    {\n      \"event_description\": \"...\",\n      \"start_time\": 0.0,\n      \"end_time\": 0.0\n    }\n  ]\n}\n"
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                # Per Gemini content-part schema, media parts expect mime_type and data at the top level
                {"type": "media", "mime_type": mime_type, "data": encoded_video},
            ]
        )

        response = self.model.invoke([message])

        # Try structured parse first
        try:
            parsed = self.parser.parse(response.content)
            return parsed.dict() if hasattr(parsed, "dict") else parsed
        except Exception:
            # Fallback: best-effort JSON extraction
            match = re.search(r"\{.*\}", str(response.content), re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            # Final fallback: wrap raw content as a single event
            return {
                "events": [
                    {
                        "event_description": str(response.content),
                        "start_time": 0.0,
                        "end_time": 0.0,
                    }
                ]
            }
