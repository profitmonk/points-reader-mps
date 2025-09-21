#!/usr/bin/env python3
"""
Simple Invoice OCR Web Application (No MCP Dependencies)
Pure FastAPI web interface for POINTS-Reader OCR
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

# Your POINTS-Reader integration
from next_steps_guide import load_local_model, patch_pointsv15_for_mps, download_local_model, pdf_to_clean_pngs, prep_existing_png

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
        self.model_loading = False
        
    async def initialize_model(self):
        """Initialize the local POINTS-Reader model"""
        if self.model_loading or self.model_loaded:
            return
            
        self.model_loading = True
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
        finally:
            self.model_loading = False
    
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
            await self.initialize_model()
        
        prompt = custom_prompt or "Extract all text from this image. Return tables in HTML format and other text in Markdown format."
        
        # Convert path to Path object
        image_path_obj = Path(image_path)
        
        # Use the updated next_steps_guide functions
        if image_path_obj.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            # For images, prepare them using prep_existing_png
            work_root = Path("work")
            pages_root = work_root / "pages"
            pages_root.mkdir(parents=True, exist_ok=True)
            
            stem = image_path_obj.stem
            processed_images = prep_existing_png(image_path_obj, pages_root, stem)
        else:
            # For PDFs, convert to images using pdf_to_clean_pngs
            work_root = Path("work")
            pages_root = work_root / "pages"
            pages_root.mkdir(parents=True, exist_ok=True)
            
            stem = image_path_obj.stem
            processed_images = pdf_to_clean_pngs(image_path_obj, pages_root, stem, dpi=300, max_pages=None)
        
        # Process the first image (or combine multiple if needed)
        if not processed_images:
            raise HTTPException(status_code=400, detail="No images could be processed from the file")
        
        # For now, process just the first image/page
        first_image = processed_images[0]
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(first_image)},
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
            logger.info(f"Created OCR_Results folder: {folder_id}")
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
            
            logger.info(f"Saved {text_filename} to Google Drive: {file['id']}")
            return file['id']
            
        finally:
            os.unlink(tmp_path)

# Initialize the service
ocr_service = InvoiceOCRService()

# Create FastAPI app
app = FastAPI(
    title="Invoice OCR Web App", 
    description="Process invoices with POINTS-Reader OCR",
    version="1.0.0"
)

# Create templates and static directories
templates_dir = Path("templates")
static_dir = Path("static")
uploads_dir = Path("uploads")

for directory in [templates_dir, static_dir, uploads_dir]:
    directory.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for jobs
processing_jobs = {}

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Invoice OCR Web Application...")
    # Model will be loaded on first use to avoid startup delays

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
    processing_jobs[job_id] = {"status": "processing", "progress": 0}
    
    # Save uploaded file
    file_path = uploads_dir / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process in background
        background_tasks.add_task(
            process_file_background, 
            job_id, 
            file_path, 
            custom_prompt, 
            file.filename or "unknown"
        )
        
        return {"job_id": job_id, "status": "processing"}
        
    except Exception as e:
        # Clean up file if something went wrong
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Check processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.post("/setup-google-drive")
async def setup_google_drive(credentials: UploadFile = File(...)):
    """Upload Google Drive credentials file"""
    if not credentials.filename or not credentials.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Please upload a JSON credentials file")
    
    # Save credentials file
    creds_path = "google_credentials.json"
    
    try:
        with open(creds_path, "wb") as buffer:
            content = await credentials.read()
            buffer.write(content)
        
        # Test the credentials by initializing Google Drive
        await ocr_service.initialize_google_drive(creds_path)
        
        return {"status": "success", "message": "Google Drive connected successfully"}
        
    except Exception as e:
        # Clean up credentials file if connection failed
        if os.path.exists(creds_path):
            os.remove(creds_path)
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
    
    background_tasks.add_task(
        process_drive_file_background, 
        job_id, 
        file_id, 
        file_name, 
        custom_prompt
    )
    
    return {"job_id": job_id, "status": "processing"}

async def process_file_background(job_id: str, file_path: Path, custom_prompt: str, original_filename: str):
    """Background task to process uploaded file"""
    try:
        processing_jobs[job_id]["progress"] = 25
        processing_jobs[job_id]["status"] = "Converting file..."
        
        # Process with your updated OCR function
        result = await ocr_service.process_image(str(file_path), custom_prompt)
        processing_jobs[job_id]["progress"] = 90
        processing_jobs[job_id]["status"] = "Finalizing..."
        
        # Clean up
        if file_path.exists():
            file_path.unlink()
        
        processing_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": result,
            "filename": original_filename
        }
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {e}")
        
        # Clean up files on error
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        
        processing_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }

async def process_drive_file_background(job_id: str, file_id: str, file_name: str, custom_prompt: str):
    """Background task to process Google Drive file"""
    temp_path = None
    
    try:
        processing_jobs[job_id]["progress"] = 10
        processing_jobs[job_id]["status"] = "Downloading from Google Drive..."
        
        # Download from Google Drive
        file_content = await ocr_service.download_drive_file(file_id)
        
        processing_jobs[job_id]["progress"] = 30
        processing_jobs[job_id]["status"] = "Converting file..."
        
        # Save temporarily
        temp_path = uploads_dir / f"temp_{job_id}_{file_name}"
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        processing_jobs[job_id]["progress"] = 50
        processing_jobs[job_id]["status"] = "Processing with OCR..."
        
        # Process with OCR using your updated function
        result = await ocr_service.process_image(str(temp_path), custom_prompt)
        
        processing_jobs[job_id]["progress"] = 90
        processing_jobs[job_id]["status"] = "Saving to Google Drive..."
        
        # Save back to Google Drive
        saved_file_id = await ocr_service.save_text_to_drive(result, file_name)
        
        # Clean up temporary files
        if temp_path and temp_path.exists():
            temp_path.unlink()
        
        processing_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": result,
            "filename": file_name,
            "saved_file_id": saved_file_id
        }
        
    except Exception as e:
        logger.error(f"Drive processing failed for job {job_id}: {e}")
        
        # Clean up files on error
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        
        processing_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ocr_service.model_loaded,
        "drive_connected": ocr_service.drive_service is not None,
        "version": "1.0.0"
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if not ocr_service.model_loaded:
        return {"status": "not_loaded", "message": "Model not yet initialized"}
    
    return {
        "status": "loaded",
        "model_name": "POINTS-Reader",
        "local_path": "./POINTS-Reader-local",
        "capabilities": ["OCR", "Document Processing", "Table Extraction"]
    }

if __name__ == "__main__":
    print("🚀 Starting Invoice OCR Web Application")
    print("📱 Web interface: http://localhost:8000")
    print("📋 API docs: http://localhost:8000/docs")
    print("💡 The model will load automatically on first use")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
