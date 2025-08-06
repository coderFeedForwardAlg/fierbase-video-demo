from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Video Vault API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://video-vault-gnytb.web.app/projects"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Video Vault API"}

@app.get("/ping")
async def ping():
    return {"status": "success", "message": "pong"}

# Add your API endpoints here

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
