#!/usr/bin/env python3
"""
Fixed Complete Invoice OCR Web Application
Uses isolated processing wrapper to prevent file conflicts
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
import io
from pathlib import Path
from typing import List, Dict, Optional
import shutil

# Web framework
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
import uvicorn

# Import the web-specific wrapper
from web_wrapper_service import web_ocr

# Google Drive integration
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

class GoogleDriveService:
    """Handles Google Drive operations separately from OCR processing"""
    
    def __init__(self):
        self.drive_service = None
    
    async def initialize_google_drive(self, credentials_path: str):
        """Initialize Google Drive API connection"""
        creds = None
        token_path = "token.json"
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GoogleRequest())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=8080)
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive API initialized")
    
    async def get_drive_pdfs(self, folder_name: str = None) -> List[Dict]:
        """Get PDF files from Google Drive"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        query = "mimeType='application/pdf'"
        if folder_name:
            folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            folder_results = self.drive_service.files().list(q=folder_query).execute()
            folders = folder_results.get('files', [])
            
            if folders:
                folder_id = folders[0]['id']
                query += f" and parents in '{folder_id}'"
        
        results = self.drive_service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, size, modifiedTime)",
            pageSize=50
        ).execute()
        
        return results.get('files', [])
    
    async def download_drive_file(self, file_id: str) -> bytes:
        """Download a file from Google Drive"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        request = self.drive_service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        return file_io.getvalue()
    
    async def save_text_to_drive(self, text_content: str, original_filename: str) -> str:
        """Save extracted text back to Google Drive"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        # Create OCR results folder
        folder_name = "OCR_Results"
        folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        folder_results = self.drive_service.files().list(q=folder_query).execute()
        folders = folder_results.get('files', [])
        
        if not folders:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive_service.files().create(body=folder_metadata).execute()
            folder_id = folder['id']
            logger.info(f"Created OCR_Results folder: {folder_id}")
        else:
            folder_id = folders[0]['id']
        
        # Create text file
        base_name = Path(original_filename).stem
        text_filename = f"{base_name}_OCR.txt"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(text_content)
            tmp_path = tmp_file.name
        
        try:
            file_metadata = {
                'name': text_filename,
                'parents': [folder_id]
            }
            
            media = MediaFileUpload(tmp_path, mimetype='text/plain')
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"Saved {text_filename} to Google Drive: {file['id']}")
            return file['id']
            
        finally:
            os.unlink(tmp_path)

# Initialize services
drive_service = GoogleDriveService()

# Create FastAPI app
app = FastAPI(
    title="Invoice OCR Web App (Fixed)", 
    description="Process invoices with isolated POINTS-Reader OCR",
    version="2.0.0"
)

# Create directories
templates_dir = Path("templates")
static_dir = Path("static")
uploads_dir = Path("uploads")

for directory in [templates_dir, static_dir, uploads_dir]:
    directory.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for jobs with better status tracking
processing_jobs = {}

@app.on_event("startup")
async def startup_event():
    """Model is already initialized in web_ocr wrapper"""
    logger.info("=== STARTING FIXED INVOICE OCR WEB APPLICATION ===")
    logger.info("Model pre-loaded via WebOCRWrapper")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface"""
    model_info = web_ocr.get_model_info()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": model_info["model_loaded"],
        "drive_connected": drive_service.drive_service is not None
    })

@app.post("/upload-image")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_prompt: str = Form("")
):
    """Upload and process a single image with isolated processing"""
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.pdf'}
    file_ext = Path(file.filename or "").suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(allowed_extensions)} files are supported"
        )
    
    # Create unique job ID
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status": f"File uploaded: {file.filename}",
        "progress": 5,
        "filename": file.filename
    }
    
    # Save uploaded file with job ID to prevent conflicts
    file_path = uploads_dir / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved upload: {file_path}")
        
        # Process in background with isolated processing
        background_tasks.add_task(
            process_file_background, 
            job_id, 
            file_path, 
            custom_prompt, 
            file.filename or "unknown"
        )
        
        return {"job_id": job_id, "status": "processing"}
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Check processing job status with detailed updates"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.post("/setup-google-drive")
async def setup_google_drive(credentials: UploadFile = File(...)):
    """Upload Google Drive credentials file"""
    if not credentials.filename or not credentials.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Please upload a JSON credentials file")
    
    creds_path = "google_credentials.json"
    
    try:
        with open(creds_path, "wb") as buffer:
            content = await credentials.read()
            buffer.write(content)
        
        await drive_service.initialize_google_drive(creds_path)
        return {"status": "success", "message": "Google Drive connected successfully"}
        
    except Exception as e:
        if os.path.exists(creds_path):
            os.remove(creds_path)
        raise HTTPException(status_code=400, detail=f"Failed to connect to Google Drive: {str(e)}")

@app.get("/drive-files")
async def list_drive_files(folder_name: str = None):
    """List PDF files from Google Drive"""
    try:
        files = await drive_service.get_drive_pdfs(folder_name)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process-drive-file")
async def process_drive_file(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    file_name: str = Form(...),
    custom_prompt: str = Form("")
):
    """Process a file from Google Drive"""
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status": f"Starting Google Drive download: {file_name}",
        "progress": 5,
        "filename": file_name
    }
    
    background_tasks.add_task(
        process_drive_file_background, 
        job_id, 
        file_id, 
        file_name, 
        custom_prompt
    )
    
    return {"job_id": job_id, "status": "processing"}

async def process_file_background(job_id: str, file_path: Path, custom_prompt: str, original_filename: str):
    """Background task with isolated processing and detailed status updates"""
    try:
        processing_jobs[job_id].update({
            "status": f"Preparing to process: {original_filename}",
            "progress": 15
        })
        
        # Use the isolated wrapper for processing
        processing_jobs[job_id].update({
            "status": "Converting and processing with OCR...",
            "progress": 30
        })
        
        # Call the isolated wrapper
        result_data = web_ocr.process_single_file(
            file_path=file_path,
            custom_prompt=custom_prompt,
            job_id=job_id
        )
        
        processing_jobs[job_id].update({
            "status": "OCR processing completed",
            "progress": 90
        })
        
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up uploaded file: {file_path}")
        
        # Update with final results
        processing_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": result_data["result"],
            "filename": original_filename,
            "pages_processed": result_data["pages_processed"]
        }
        
        logger.info(f"Processing completed successfully for job {job_id}")
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {e}")
        
        # Clean up on error
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        
        processing_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e),
            "filename": original_filename
        }

async def process_drive_file_background(job_id: str, file_id: str, file_name: str, custom_prompt: str):
    """Background task for Google Drive files with isolated processing"""
    temp_path = None
    
    try:
        processing_jobs[job_id].update({
            "status": f"Downloading from Google Drive: {file_name}",
            "progress": 10
        })
        
        # Download from Google Drive
        file_content = await drive_service.download_drive_file(file_id)
        
        processing_jobs[job_id].update({
            "status": "File downloaded, preparing for OCR...",
            "progress": 25
        })
        
        # Save temporarily with job-specific name
        temp_path = uploads_dir / f"drive_{job_id}_{file_name}"
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        processing_jobs[job_id].update({
            "status": "Processing with OCR...",
            "progress": 40
        })
        
        # Use isolated wrapper for processing
        result_data = web_ocr.process_single_file(
            file_path=temp_path,
            custom_prompt=custom_prompt,
            job_id=job_id
        )
        
        processing_jobs[job_id].update({
            "status": "Saving results to Google Drive...",
            "progress": 80
        })
        
        # Save back to Google Drive
        saved_file_id = await drive_service.save_text_to_drive(result_data["result"], file_name)
        
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()
            logger.info(f"Cleaned up temp file: {temp_path}")
        
        processing_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": result_data["result"],
            "filename": file_name,
            "saved_file_id": saved_file_id,
            "pages_processed": result_data["pages_processed"]
        }
        
        logger.info(f"Drive processing completed successfully for job {job_id}")
        
    except Exception as e:
        logger.error(f"Drive processing failed for job {job_id}: {e}")
        
        # Clean up on error
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        
        processing_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e),
            "filename": file_name
        }

@app.get("/health")
async def health_check():
    """Health check with model status"""
    model_info = web_ocr.get_model_info()
    return {
        "status": "healthy",
        "model_loaded": model_info["model_loaded"],
        "device": model_info["device"],
        "drive_connected": drive_service.drive_service is not None,
        "version": "2.0.0 (Fixed)"
    }

@app.post("/cleanup")
async def cleanup_old_jobs():
    """Manual cleanup endpoint for maintenance"""
    web_ocr.cleanup_old_jobs(max_age_hours=24)
    return {"status": "cleanup completed"}

if __name__ == "__main__":
    print("🚀 Starting Fixed Invoice OCR Web Application")
    print("📱 Web interface: http://localhost:8000")
    print("🔧 Using isolated processing to prevent file conflicts")
    print("📋 API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
