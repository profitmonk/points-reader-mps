#!/usr/bin/env python3
"""
MCP Invoice OCR Web Application
Combines MCP server functionality with a FastAPI web interface
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
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

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import mcp.types as types

# Your POINTS-Reader integration
from next_steps_guide import load_local_model, patch_pointsv15_for_mps, download_local_model, ensure_image

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

class InvoiceOCRService:
    """Core service that handles OCR processing and Google Drive operations"""
    
    def __init__(self):
        self.drive_service = None
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_loaded = False
        
        # Initialize local POINTS-Reader model
        asyncio.create_task(self.initialize_model())
    
    async def initialize_model(self):
        """Initialize the local POINTS-Reader model"""
        try:
            logger.info("Initializing POINTS-Reader model...")
            model_dir = download_local_model(Path("POINTS-Reader-local"))
            patch_pointsv15_for_mps(model_dir)
            self.model, self.tokenizer, self.image_processor = load_local_model(Path("POINTS-Reader-local"))
            self.model_loaded = True
            logger.info("POINTS-Reader model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
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
                creds = flow.run_local_server(port=0)
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive API initialized")
    
    async def process_image(self, image_path: str, custom_prompt: str = None) -> str:
        """Process an image with POINTS-Reader OCR"""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        prompt = custom_prompt or "Extract all text from this image. Return tables in HTML format and other text in Markdown format."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]
        
        import torch
        with torch.inference_mode():
            result = self.model.chat(
                messages, 
                self.tokenizer, 
                self.image_processor, 
                {"max_new_tokens": 2048, "temperature": 0.0}
            )
        
        return result
    
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
            fields="nextPageToken, files(id, name, size, modifiedTime)"
        ).execute()
        
        return results.get('files', [])
    
    async def save_text_to_drive(self, text_content: str, original_filename: str) -> str:
        """Save extracted text back to Google Drive"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        # Create OCR results folder if it doesn't exist
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
        else:
            folder_id = folders[0]['id']
        
        # Create text file name
        base_name = Path(original_filename).stem
        text_filename = f"{base_name}_OCR.txt"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(text_content)
            tmp_path = tmp_file.name
        
        try:
            # Upload to Google Drive
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
            
            return file['id']
            
        finally:
            os.unlink(tmp_path)

# Initialize the service
ocr_service = InvoiceOCRService()

# Create FastAPI app
app = FastAPI(title="Invoice OCR Web App", description="Process invoices with POINTS-Reader OCR")

# Create templates and static directories
templates_dir = Path("templates")
static_dir = Path("static")
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for jobs
processing_jobs = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": ocr_service.model_loaded,
        "drive_connected": ocr_service.drive_service is not None
    })

@app.post("/upload-image")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_prompt: str = Form("")
):
    """Upload and process a single image"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, and PDF files are supported")
    
    # Create unique job ID
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {"status": "processing", "progress": 0}
    
    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{job_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    background_tasks.add_task(process_file_background, job_id, file_path, custom_prompt, file.filename)
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Check processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.post("/setup-google-drive")
async def setup_google_drive(credentials: UploadFile = File(...)):
    """Upload Google Drive credentials file"""
    if not credentials.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Please upload a JSON credentials file")
    
    # Save credentials file
    creds_path = "google_credentials.json"
    with open(creds_path, "wb") as buffer:
        shutil.copyfileobj(credentials.file, buffer)
    
    try:
        await ocr_service.initialize_google_drive(creds_path)
        return {"status": "success", "message": "Google Drive connected successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to Google Drive: {str(e)}")

@app.get("/drive-files")
async def list_drive_files(folder_name: str = None):
    """List PDF files from Google Drive"""
    try:
        files = await ocr_service.get_drive_pdfs(folder_name)
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
    processing_jobs[job_id] = {"status": "processing", "progress": 0}
    
    background_tasks.add_task(process_drive_file_background, job_id, file_id, file_name, custom_prompt)
    
    return {"job_id": job_id, "status": "processing"}

async def process_file_background(job_id: str, file_path: Path, custom_prompt: str, original_filename: str):
    """Background task to process uploaded file"""
    try:
        processing_jobs[job_id]["progress"] = 25
        
        # Convert PDF to image if needed
        image_path = ensure_image(file_path)
        processing_jobs[job_id]["progress"] = 50
        
        # Process with OCR
        result = await ocr_service.process_image(str(image_path), custom_prompt)
        processing_jobs[job_id]["progress"] = 90
        
        # Clean up
        if file_path != image_path:
            os.unlink(image_path)
        os.unlink(file_path)
        
        processing_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": result,
            "filename": original_filename
        }
        
    except Exception as e:
        processing_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }

async def process_drive_file_background(job_id: str, file_id: str, file_name: str, custom_prompt: str):
    """Background task to process Google Drive file"""
    try:
        processing_jobs[job_id]["progress"] = 10
        
        # Download from Google Drive
        request = ocr_service.drive_service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            processing_jobs[job_id]["progress"] = 10 + (status.progress() * 40)
        
        # Save temporarily
        temp_path = Path(f"temp_{job_id}_{file_name}")
        with open(temp_path, 'wb') as f:
            f.write(file_io.getvalue())
        
        processing_jobs[job_id]["progress"] = 60
        
        # Convert to image if PDF
        image_path = ensure_image(temp_path)
        processing_jobs[job_id]["progress"] = 70
        
        # Process with OCR
        result = await ocr_service.process_image(str(image_path), custom_prompt)
        processing_jobs[job_id]["progress"] = 90
        
        # Save back to Google Drive
        saved_file_id = await ocr_service.save_text_to_drive(result, file_name)
        
        # Clean up
        if temp_path != image_path:
            os.unlink(image_path)
        os.unlink(temp_path)
        
        processing_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": result,
            "filename": file_name,
            "saved_file_id": saved_file_id
        }
        
    except Exception as e:
        processing_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ocr_service.model_loaded,
        "drive_connected": ocr_service.drive_service is not None
    }

if __name__ == "__main__":
    print("🚀 Starting Invoice OCR Web Application")
    print("📱 Web interface: http://localhost:8000")
    print("📋 API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
