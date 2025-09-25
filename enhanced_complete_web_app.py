#!/usr/bin/env python3
"""
Enhanced Complete Invoice OCR Web Application
- Fixed shared folder access
- Added folder selection and batch processing
- Improved error handling and status updates
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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
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
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file', 
    'https://www.googleapis.com/auth/drive.metadata.readonly'
]

class EnhancedGoogleDriveService:
    """Enhanced Google Drive service with shared folder support and batch processing"""
    
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
                # Fixed: Use consistent port and handle trailing slash
                creds = flow.run_local_server(port=8080, open_browser=True)
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive API initialized")
    
    async def get_folders(self, include_shared: bool = True) -> List[Dict]:
        """Get all folders from Google Drive including shared ones"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        query = "mimeType='application/vnd.google-apps.folder'"
        if include_shared:
            query += " and (sharedWithMe=true or 'me' in owners)"
        
        results = self.drive_service.files().list(
            q=query,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields="nextPageToken, files(id, name, owners, shared, parents)",
            pageSize=100,
            orderBy="name"
        ).execute()
        
        folders = results.get('files', [])
        
        # Add root folder option
        root_folder = {
            'id': 'root',
            'name': 'My Drive (Root)',
            'owners': [{'me': True}],
            'shared': False
        }
        
        return [root_folder] + folders
    
    async def get_pdfs_in_folder(self, folder_id: str = None, include_shared: bool = True) -> List[Dict]:
        """Get all PDF files in a specific folder - FIXED for shared folders with proper fields"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        try:
            query = "mimeType='application/pdf'"
            if folder_id and folder_id != 'root':
                query += f" and parents in '{folder_id}'"
            elif folder_id == 'root':
                query += " and 'root' in parents"
            
            logger.info(f"DEBUG: Searching folder {folder_id} with query: {query}")
            
            # Get ALL PDFs with proper pagination
            all_files = []
            page_token = None
            
            while True:
                results = self.drive_service.files().list(
                    q=query,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    # FIXED: Request proper fields including size
                    fields="nextPageToken, files(id, name, size, modifiedTime, owners, shared, webViewLink, mimeType)",
                    pageSize=100,  # FIXED: Increased from debug limit of 10
                    orderBy="name",
                    pageToken=page_token
                ).execute()
                
                files_batch = results.get('files', [])
                all_files.extend(files_batch)
                
                # Check for more pages
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            logger.info(f"DEBUG: Found {len(all_files)} PDF files total (after pagination)")
            
            # Log first few files with size info
            for i, file in enumerate(all_files[:3]):
                size = file.get('size', 'Unknown')
                logger.info(f"DEBUG: File {i+1}: {file.get('name')} | Size: {size} bytes")
            
            return all_files
            
        except Exception as e:
            logger.error(f"DEBUG: API Error for folder {folder_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Drive API error: {str(e)}")

    async def get_drive_pdfs(self, folder_name: str = None, include_shared: bool = True) -> List[Dict]:
        """Get PDF files from Google Drive (legacy method for compatibility)"""
        if folder_name:
            # Find folder by name
            folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if include_shared:
                folder_query += " and (sharedWithMe=true or 'me' in owners)"
            
            folder_results = self.drive_service.files().list(
                q=folder_query,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            folders = folder_results.get('files', [])
            
            if folders:
                return await self.get_pdfs_in_folder(folders[0]['id'], include_shared)
            else:
                return []
        else:
            return await self.get_pdfs_in_folder(None, include_shared)
    
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
    
    async def save_text_to_drive(self, text_content: str, original_filename: str, folder_id: str = None) -> str:
        """Save extracted text back to Google Drive"""
        if not self.drive_service:
            raise HTTPException(status_code=400, detail="Google Drive not initialized")
        
        # Create OCR results folder
        folder_name = "OCR_Results"
        parent_folder_id = folder_id or 'root'
        
        # Check if OCR_Results folder exists in the target location
        folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and parents in '{parent_folder_id}'"
        folder_results = self.drive_service.files().list(q=folder_query).execute()
        folders = folder_results.get('files', [])
        
        if not folders:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = self.drive_service.files().create(body=folder_metadata).execute()
            results_folder_id = folder['id']
            logger.info(f"Created OCR_Results folder: {results_folder_id}")
        else:
            results_folder_id = folders[0]['id']
        
        # Create text file
        base_name = Path(original_filename).stem
        text_filename = f"{base_name}_OCR.txt"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(text_content)
            tmp_path = tmp_file.name
        
        try:
            file_metadata = {
                'name': text_filename,
                'parents': [results_folder_id]
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
drive_service = EnhancedGoogleDriveService()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Invoice OCR Web App", 
    description="Process invoices with folder selection and batch processing",
    version="2.1.0"
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
batch_jobs = {}  # Separate tracking for batch jobs

@app.on_event("startup")
async def startup_event():
    """Model is already initialized in web_ocr wrapper"""
    logger.info("=== STARTING ENHANCED INVOICE OCR WEB APPLICATION ===")
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
        "filename": file.filename,
        "type": "single"
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
    if job_id in processing_jobs:
        return processing_jobs[job_id]
    elif job_id in batch_jobs:
        return batch_jobs[job_id]
    else:
        raise HTTPException(status_code=404, detail="Job not found")

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

@app.get("/drive-folders")
async def list_drive_folders(include_shared: bool = Query(True)):
    """List folders from Google Drive including shared ones"""
    try:
        folders = await drive_service.get_folders(include_shared)
        return {"folders": folders}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drive-files")
async def list_drive_files(
    folder_id: str = Query(None), 
    folder_name: str = Query(None),
    include_shared: bool = Query(True)
):
    """List PDF files from Google Drive folder"""
    try:
        if folder_id:
            files = await drive_service.get_pdfs_in_folder(folder_id, include_shared)
        else:
            files = await drive_service.get_drive_pdfs(folder_name, include_shared)
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
    """Process a single file from Google Drive"""
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status": f"Starting Google Drive download: {file_name}",
        "progress": 5,
        "filename": file_name,
        "type": "single"
    }
    
    background_tasks.add_task(
        process_drive_file_background, 
        job_id, 
        file_id, 
        file_name, 
        custom_prompt
    )
    
    return {"job_id": job_id, "status": "processing"}

@app.post("/process-drive-folder")
async def process_drive_folder(
    background_tasks: BackgroundTasks,
    folder_id: str = Form(...),
    folder_name: str = Form(...),
    custom_prompt: str = Form("")
):
    """Process all PDF files in a Google Drive folder"""
    batch_job_id = str(uuid.uuid4())
    
    try:
        # Get all PDFs in the folder
        files = await drive_service.get_pdfs_in_folder(folder_id, include_shared=True)
        
        if not files:
            raise HTTPException(status_code=400, detail=f"No PDF files found in folder '{folder_name}'")
        
        batch_jobs[batch_job_id] = {
            "status": f"Found {len(files)} PDF files in '{folder_name}'",
            "progress": 5,
            "folder_name": folder_name,
            "total_files": len(files),
            "processed_files": 0,
            "type": "batch",
            "results": [],
            "errors": []
        }
        
        # Process all files in background
        background_tasks.add_task(
            process_folder_background,
            batch_job_id,
            files,
            custom_prompt,
            folder_name
        )
        
        return {"job_id": batch_job_id, "status": "processing", "total_files": len(files)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")

# Background processing functions
async def process_file_background(job_id: str, file_path: Path, custom_prompt: str, original_filename: str):
    """Background task with isolated processing and detailed status updates"""
    try:
        processing_jobs[job_id].update({
            "status": f"Preparing to process: {original_filename}",
            "progress": 15
        })
        
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
            "pages_processed": result_data["pages_processed"],
            "type": "single"
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
            "filename": original_filename,
            "type": "single"
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
            "pages_processed": result_data["pages_processed"],
            "type": "single"
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
            "filename": file_name,
            "type": "single"
        }

async def process_folder_background(batch_job_id: str, files: List[Dict], custom_prompt: str, folder_name: str):
    """Background task for processing all files in a folder"""
    total_files = len(files)
    processed_files = 0
    results = []
    errors = []
    
    try:
        for i, file_info in enumerate(files, 1):
            file_id = file_info['id']
            file_name = file_info['name']
            
            batch_jobs[batch_job_id].update({
                "status": f"Processing {file_name} ({i}/{total_files})",
                "progress": int((i / total_files) * 80),  # Reserve 80% for processing
                "processed_files": processed_files
            })
            
            temp_path = None
            try:
                # Download file
                file_content = await drive_service.download_drive_file(file_id)
                temp_path = uploads_dir / f"batch_{batch_job_id}_{i}_{file_name}"
                
                with open(temp_path, 'wb') as f:
                    f.write(file_content)
                
                # Process with OCR
                result_data = web_ocr.process_single_file(
                    file_path=temp_path,
                    custom_prompt=custom_prompt,
                    job_id=f"{batch_job_id}_{i}"
                )
                
                # Save result back to Drive
                saved_file_id = await drive_service.save_text_to_drive(
                    result_data["result"], 
                    file_name
                )
                
                # Add to results
                results.append({
                    "filename": file_name,
                    "result": result_data["result"],
                    "pages_processed": result_data["pages_processed"],
                    "saved_file_id": saved_file_id
                })
                
                processed_files += 1
                
                # Clean up temp file
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                
                logger.info(f"Batch: Successfully processed {file_name} ({i}/{total_files})")
                
            except Exception as e:
                error_msg = f"Failed to process {file_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Batch processing error: {error_msg}")
                
                # Clean up temp file on error
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
        
        # Final update
        batch_jobs[batch_job_id] = {
            "status": "completed",
            "progress": 100,
            "folder_name": folder_name,
            "total_files": total_files,
            "processed_files": processed_files,
            "successful": len(results),
            "failed": len(errors),
            "type": "batch",
            "results": results,
            "errors": errors
        }
        
        logger.info(f"Batch processing completed: {processed_files}/{total_files} files processed successfully")
        
    except Exception as e:
        batch_jobs[batch_job_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e),
            "folder_name": folder_name,
            "total_files": total_files,
            "processed_files": processed_files,
            "type": "batch"
        }
        logger.error(f"Batch processing failed: {e}")

@app.get("/health")
async def health_check():
    """Health check with model status"""
    model_info = web_ocr.get_model_info()
    return {
        "status": "healthy",
        "model_loaded": model_info["model_loaded"],
        "device": model_info["device"],
        "drive_connected": drive_service.drive_service is not None,
        "version": "2.1.0 (Enhanced)"
    }

@app.post("/cleanup")
async def cleanup_old_jobs():
    """Manual cleanup endpoint for maintenance"""
    web_ocr.cleanup_old_jobs(max_age_hours=24)
    return {"status": "cleanup completed"}

if __name__ == "__main__":
    print("🚀 Starting Enhanced Invoice OCR Web Application")
    print("📱 Web interface: http://localhost:8000")
    print("🔧 Features: Folder selection, batch processing, shared drive support")
    print("📋 API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
