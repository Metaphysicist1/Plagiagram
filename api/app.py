import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import sys

# Add app directory to Python path
sys.path.append('/app')

from models.plagiarism_detector import PlagiarismDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Code Plagiarism Detector API",
    description="API for detecting code plagiarism using vector similarity and LLM analysis",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Initialize plagiarism detector
detector = PlagiarismDetector()

class CodeSubmission(BaseModel):
    """Pydantic model for code submission."""
    code: str
    language: str = "Unknown"
    file_name: Optional[str] = None

class PlagiarismResponse(BaseModel):
    """Pydantic model for plagiarism detection response."""
    is_plagiarized: bool
    explanation: str
    detailed_analysis: Optional[str] = None
    matches: Optional[list] = None
    evaluation_file: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def index():
    """Serve the main HTML page."""
    return FileResponse("api/static/index.html")

@app.post("/detect", response_model=PlagiarismResponse)
async def detect_plagiarism(submission: CodeSubmission):
    """
    Detect plagiarism in submitted code.
    
    Args:
        submission: CodeSubmission object containing code and language
        
    Returns:
        PlagiarismResponse object with detection results
    """
    try:
        # Get code and language from submission
        code = submission.code
        language = submission.language
        file_name = submission.file_name or "unknown_file"
        
        # Detect plagiarism
        result = detector.detect_plagiarism(
            code_vector=code,  # Note: This should be pre-computed vector in production
            language=language,
            code=code,
            query_file=file_name
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 


    