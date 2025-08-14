import os
from typing import Optional, Tuple, Any, Dict, TypedDict, Annotated
from flask import Flask, jsonify, request
import subprocess
import tempfile
from pathlib import Path
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    AnyMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from pydantic import BaseModel, Field

# Firebase Admin for verifying ID tokens
import firebase_admin
from firebase_admin import auth as fb_auth

# Google Cloud Storage for listing video objects
from google.cloud import storage

app = Flask(__name__)

# Initialize Firebase Admin SDK once per process. On Cloud Run, this uses
# the service account attached to the service automatically.
if not firebase_admin._apps:
    firebase_admin.initialize_app()

# Reuse a single Storage client
_storage_client = storage.Client()


# Lazy-initialized Gemini analyzer
_analyzer_instance = None


def _get_analyzer():
    """Create and cache a Gemini analyzer if GEMINI_API_KEY is configured.

    Returns None if the analyzer cannot be created (e.g., missing key), allowing
    the API to continue returning filenames without analysis rather than failing.
    """
    global _analyzer_instance
    if _analyzer_instance is not None:
        return _analyzer_instance
    try:
        # Local import to avoid ImportError when dependency is not installed yet
        from gemini_analyzer import GeminiVideoAnalyzer  # type: ignore

        _analyzer_instance = GeminiVideoAnalyzer()
        return _analyzer_instance
    except Exception:
        # Analyzer unavailable (likely no GEMINI_API_KEY or missing deps)
        return None


def _summarize_events(events: Any) -> str:
    """Create a concise single-line description from events list."""
    try:
        parts = []
        for e in events or []:
            desc = e.get("event_description") if isinstance(e, dict) else getattr(e, "event_description", None)
            start = e.get("start_time") if isinstance(e, dict) else getattr(e, "start_time", None)
            end = e.get("end_time") if isinstance(e, dict) else getattr(e, "end_time", None)
            if desc is None:
                continue
            if start is not None and end is not None:
                parts.append(f"[{start:.2f}-{end:.2f}s] {desc}")
            else:
                parts.append(str(desc))
        return "; ".join(parts)[:1000]
    except Exception:
        return ""


def _get_project_id() -> Optional[str]:
    # Prefer the project detected by the Google Cloud client
    try:
        if _storage_client.project:
            return _storage_client.project
    except Exception:
        pass
    # Fallback to common environment variables
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("FIREBASE_CONFIG_PROJECT_ID")
    )


def _get_bucket_name() -> str:
    """
    Determine the storage bucket without requiring an env var.
    Order of precedence:
      1) Explicit env var BUCKET_NAME or GCS_BUCKET (if provided)
      2) FIREBASE_STORAGE_BUCKET if present
      3) Detect project ID, then use the default Firebase bucket naming convention:
         <project-id>.appspot.com
    """
    # 1) explicit overrides
    explicit = os.environ.get("BUCKET_NAME") or os.environ.get("GCS_BUCKET")
    if explicit:
        return explicit
    # 2) Firebase-style env var if available
    fb_bucket = os.environ.get("FIREBASE_STORAGE_BUCKET")
    if fb_bucket:
        return fb_bucket

    project_id = _get_project_id()
    if not project_id:
        raise RuntimeError(
            "Unable to resolve project ID to infer bucket name. Set GOOGLE_CLOUD_PROJECT or BUCKET_NAME."
        )

    name = f"{project_id}.appspot.com"
    try:
        b = _storage_client.bucket(name)
        # If we have permission to check, return upon confirmation.
        if b.exists():
            return name
    except Exception:
        pass
    # Default to the conventional name even if we cannot check existence.
    return name


def _verify_bearer_token(req) -> Tuple[Optional[str], Optional[str]]:
    """
    Verifies Firebase ID token from Authorization header and returns (uid, error_message).
    """
    auth_header = req.headers.get("Authorization", "")
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        id_token = parts[1]
    else:
        return None, "missing_bearer_token"

    try:
        decoded = fb_auth.verify_id_token(id_token)
        uid = decoded.get("uid")
        if not uid:
            return None, "uid_missing_in_token"
        return uid, None
    except Exception as e:  # noqa: BLE001 (broad for auth failures)
        return None, f"invalid_token: {e}"


@app.get("/health")
def health():
    # Simple plaintext health check endpoint
    return "healthy", 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/")
def root():
    return jsonify({"status": "ok", "endpoints": ["/health"]})

def _collect_project_videos(
    uid: str,
    project_id: str,
    page_size: int,
    page_token: Optional[str],
    analyze_flag: bool,
    max_analyze: int,
    max_size_bytes: int,
) -> Tuple[list[Dict[str, Any]], Optional[str]]:
    """
    Core worker that lists project videos from GCS and optionally analyzes them.

    Returns (videos, next_page_token).
    """
    bucket_name = _get_bucket_name()
    bucket = _storage_client.bucket(bucket_name)

    prefixes = [
        f"users/{uid}/projects/{project_id}/videos/",
        f"users/{uid}/projects/{project_id}/media/",
    ]

    collected: list[Dict[str, Any]] = []
    remaining = page_size
    next_page_token: Optional[str] = None

    video_exts = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}

    analyzer = _get_analyzer() if analyze_flag else None
    analyzed_count = 0

    for i, pfx in enumerate(prefixes):
        if remaining <= 0:
            break
        iterator = _storage_client.list_blobs(
            bucket_or_name=bucket,
            prefix=pfx,
            max_results=remaining,
            page_token=page_token if i == 0 else None,
        )
        blobs = list(iterator)
        if i == 0:
            next_page_token = iterator.next_page_token if hasattr(iterator, "next_page_token") else None

        for b in blobs:
            name_only = b.name.split("/")[-1]
            if not name_only:
                continue
            ctype = getattr(b, "content_type", None) or ""
            if not ctype:
                lower = name_only.lower()
                if lower.endswith(".mp4"):
                    ctype = "video/mp4"
                elif lower.endswith(".webm"):
                    ctype = "video/webm"
                elif lower.endswith(".mov"):
                    ctype = "video/quicktime"
                elif lower.endswith(".mkv"):
                    ctype = "video/x-matroska"
                elif lower.endswith(".avi"):
                    ctype = "video/x-msvideo"
                elif lower.endswith(".m4v"):
                    ctype = "video/x-m4v"
            is_video = ctype.startswith("video/") or any(name_only.lower().endswith(ext) for ext in video_exts)
            if not is_video:
                continue
            item: Dict[str, Any] = {
                "name": name_only,
                "path": b.name,
                "size": getattr(b, "size", None),
                "updated": getattr(b, "updated", None).isoformat() if getattr(b, "updated", None) else None,
                "contentType": ctype or None,
            }

            # Optionally analyze the video to produce descriptions
            if analyzer is not None and analyzed_count < max_analyze:
                blob_size = getattr(b, "size", None)
                if blob_size is not None and blob_size <= max_size_bytes:
                    try:
                        data = b.download_as_bytes()
                        analysis = analyzer.analyze_video_bytes(data, mime_type=ctype or "video/mp4")
                        events = analysis.get("events") if isinstance(analysis, dict) else None
                        item["events"] = events
                        item["description"] = _summarize_events(events)
                        analyzed_count += 1
                    except Exception as analysis_err:  # noqa: BLE001
                        err_str = str(analysis_err)
                        if len(err_str) > 500:
                            err_str = err_str[:500] + "…"
                        item["analysisError"] = err_str
                else:
                    item["analysisSkipped"] = "file_too_large"

            collected.append(item)
            remaining -= 1
            if remaining <= 0:
                break

    return collected, next_page_token

@app.get("/api/projects/<project_id>/videos")
def list_videos(project_id: str):
    """
    Lists video objects for the authenticated user under:
    users/{uid}/projects/{project_id}/videos/

    Query params:
      - pageSize: optional int (default 1000, max 1000)
      - pageToken: optional str for pagination
    """
    uid, err = _verify_bearer_token(request)
    if err:
        return jsonify({"error": err}), 401

    try:
        page_size = min(int(request.args.get("pageSize", 1000)), 1000)
        page_token = request.args.get("pageToken")

        analyze_flag = request.args.get("analyze", "true").lower() != "false"
        try:
            max_analyze = max(0, min(10, int(request.args.get("maxAnalyze", 3))))
        except Exception:
            max_analyze = 3

        # Default analysis size limit (can be overridden by env var or query param)
        default_limit = int(os.environ.get("MAX_ANALYZE_BYTES", str(60 * 1024 * 1024)))  # default 60MB
        max_size_bytes = default_limit
        override_bytes = request.args.get("maxBytes")
        if override_bytes is not None:
            try:
                cap = 150 * 1024 * 1024  # 150MB cap
                max_size_bytes = max(0, min(cap, int(override_bytes)))
            except Exception:
                pass

        videos, next_page_token = _collect_project_videos(
            uid=uid,
            project_id=project_id,
            page_size=page_size,
            page_token=page_token,
            analyze_flag=analyze_flag,
            max_analyze=max_analyze,
            max_size_bytes=max_size_bytes,
        )
        return jsonify({"videos": videos, "nextPageToken": next_page_token})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "internal", "message": str(e)}), 500

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


_chat_llm = None
_chat_graph = None


def _get_chat_llm():
    global _chat_llm
    if _chat_llm is not None:
        return _chat_llm
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for chatbot.")
    model_name = os.environ.get("CHAT_MODEL", "gemini-1.5-flash")
    _chat_llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    return _chat_llm


# Tool schema and wrapper around _cut_video so the LLM can call it
class CutVideoInput(BaseModel):
    """Inputs for the cut_video tool.

    - video: A video identifier. Accepts:
        • A GCS URI like gs://bucket/path/to/file.mp4
        • A blob path within the default bucket like users/uid/projects/pid/videos/file.mp4
        • A local filesystem path (for testing)
    - start: Start time in seconds (inclusive).
    - end: End time in seconds (exclusive), must be greater than start.
    - new_video_name: File name for the output clip (extension auto-added if missing).
    """

    video: str = Field(..., description="Input video path or gs:// URI")
    start: float = Field(..., ge=0, description="Start time in seconds (inclusive)")
    end: float = Field(..., gt=0, description="End time in seconds (exclusive), must be > start")
    new_video_name: str = Field(..., description="Name for the new clip; extension auto-added if missing")


@tool("cut_video", args_schema=CutVideoInput)
def cut_video_tool(video: str, start: float, end: float, new_video_name: str) -> str:
    """Cut a segment from a video using ffmpeg and upload the result to GCS.

    Returns a JSON string with minimal metadata about the uploaded clip,
    including bucket, blob, and contentType. On error, returns a JSON
    object with an "error" field.
    """
    try:
        result = _cut_video(video=video, start=start, end=end, new_video_name=new_video_name)
        return json.dumps(result)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if len(msg) > 500:
            msg = msg[:500] + "…"
        return json.dumps({"error": msg})


def _get_chat_graph():
    global _chat_graph
    if _chat_graph is not None:
        return _chat_graph

    llm = _get_chat_llm()

    # Bind tools so the model can decide to call them
    tools = [cut_video_tool]
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: ChatState):
        # Ask the model; if it wants to call a tool, the response will include tool calls
        resp = llm_with_tools.invoke(state["messages"])  # AIMessage with optional tool_calls
        return {"messages": [resp]}

    graph = StateGraph(ChatState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    # Route to tools if there are tool calls; otherwise end
    graph.add_conditional_edges(
        "model",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph.add_edge("tools", "model")
    _chat_graph = graph.compile()
    return _chat_graph


def _coerce_messages(payload: Any) -> list[AnyMessage]:
    """Convert request JSON into LangChain message objects."""
    msgs = []
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            for m in payload["messages"]:
                if isinstance(m, dict):
                    role = (m.get("role") or "").lower()
                    content = m.get("content")
                    if not isinstance(content, str):
                        content = str(content)
                    if role in ("user", "human"):
                        msgs.append(HumanMessage(content=content))
                    elif role in ("assistant", "ai", "bot"):
                        msgs.append(AIMessage(content=content))
                    elif role == "system":
                        msgs.append(SystemMessage(content=content))
                    else:
                        msgs.append(HumanMessage(content=content))
                elif isinstance(m, str):
                    msgs.append(HumanMessage(content=m))
        elif isinstance(payload.get("message"), str):
            msgs.append(HumanMessage(content=payload["message"]))
    elif isinstance(payload, str):
        msgs.append(HumanMessage(content=payload))
    return msgs


def _messages_to_dicts(msgs: list[AnyMessage]):
    out = []
    for m in msgs:
        role = "assistant" if isinstance(m, AIMessage) else ("system" if isinstance(m, SystemMessage) else "user")
        out.append({"role": role, "content": getattr(m, "content", "")})
    return out


@app.post("/api/chatbot")
def chatbot():
    """Simple LangGraph-powered chat endpoint (no memory/persistence)."""
    uid, err = _verify_bearer_token(request)
    if err:
        return jsonify({"error": err}), 401

    try:
        data = request.get_json(silent=True) or {}
        msgs = _coerce_messages(data)
        if not msgs:
            return jsonify({"error": "invalid_request", "message": "Provide 'message' or 'messages'."}), 400

        graph = _get_chat_graph()
        state_in: ChatState = {"messages": msgs}
        state_out: ChatState = graph.invoke(state_in)
        all_msgs = state_out.get("messages", [])

        reply = ""
        for m in reversed(all_msgs):
            if isinstance(m, AIMessage):
                reply = m.content
                break

        return jsonify({
            "uid": uid,
            "reply": reply,
            "messages": _messages_to_dicts(all_msgs),
        })
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "internal", "message": str(e)}), 500

def exicut_insturction():
    """
    do what the user asked
    """
    #use langchain 


def _cut_video(video: str, start: int, end: int, new_video_name: str) -> Dict[str, Any]:
    """
    Extract a clip between [start, end) seconds from the input video using ffmpeg,
    then upload the resulting file to Google Cloud Storage.

    'video' may be:
      - A GCS URI like 'gs://bucket/path/to/file.mp4'
      - A blob path within the default bucket like 'users/uid/projects/pid/videos/file.mp4'
      - A local filesystem path (useful for testing)

    The new object will be uploaded next to the source (same folder) if the source
    is in GCS; otherwise it will be uploaded to the root of the default bucket.
    Returns a dict with minimal metadata.
    """
    # Validate time range
    start_s = float(start)
    end_s = float(end)
    if start_s < 0 or end_s <= start_s:
        raise ValueError("Invalid time range: end must be greater than start and both non-negative.")

    # Determine source and destination
    src_bucket = None
    dest_bucket = None
    src_blob_name = None
    dest_blob_name = new_video_name.strip()

    # Ensure output has an extension
    if not dest_blob_name.lower().endswith((".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")):
        dest_blob_name += ".mp4"

    local_input: Optional[str] = None
    temp_input: Optional[str] = None

    if video.startswith("gs://"):
        # gs://bucket/blob...
        path_after = video[5:]
        parts = path_after.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid GCS URI: {video}")
        bucket_name_in, src_blob_name = parts[0], parts[1]
        src_bucket = _storage_client.bucket(bucket_name_in)
        dest_bucket = src_bucket
    else:
        # Is it a local file?
        if os.path.exists(video):
            local_input = video
        else:
            # Treat as blob name within default bucket
            src_blob_name = video.lstrip("/")
            bucket_name_in = _get_bucket_name()
            src_bucket = _storage_client.bucket(bucket_name_in)
            dest_bucket = src_bucket

    # Place the new object next to the source if we have a blob path
    parent_dir = ""
    if src_blob_name:
        parent_dir = os.path.dirname(src_blob_name)
        if parent_dir:
            dest_blob_name = f"{parent_dir.rstrip('/')}/{os.path.basename(dest_blob_name)}"

    try:
        # Download input to a temp file if needed
        if local_input is None:
            if not (src_bucket and src_blob_name):
                raise ValueError("Source video must be a local path or a valid GCS blob/URI.")
            with tempfile.NamedTemporaryFile(prefix="in_", suffix=Path(src_blob_name).suffix or ".mp4", delete=False) as t_in:
                temp_input = t_in.name
            blob = src_bucket.blob(src_blob_name)
            blob.download_to_filename(temp_input)
            input_path = temp_input
        else:
            input_path = local_input

        # Prepare output temp file
        out_suffix = Path(dest_blob_name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(prefix="out_", suffix=out_suffix, delete=False) as t_out:
            output_path = t_out.name

        # Build and run ffmpeg
        duration = end_s - start_s
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_s),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if proc.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            # Include tail of stderr for easier debugging, but avoid huge payloads
            tail = (proc.stderr or "")
            if len(tail) > 2000:
                tail = tail[-2000:]
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {tail}")

        # Upload to GCS
        if dest_bucket is None:
            dest_bucket = _storage_client.bucket(_get_bucket_name())
        out_blob = dest_bucket.blob(dest_blob_name)
        content_type = "video/mp4" if out_suffix.lower() == ".mp4" else "application/octet-stream"
        out_blob.upload_from_filename(output_path, content_type=content_type)

        return {
            "bucket": dest_bucket.name,
            "blob": dest_blob_name,
            "contentType": content_type,
        }
    finally:
        # Cleanup temp files
        try:
            if temp_input and os.path.exists(temp_input):
                os.remove(temp_input)
        except Exception:
            pass
        try:
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass


def create_app():
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
