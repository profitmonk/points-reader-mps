#!/usr/bin/env python3
"""
Debug Web App - Minimal version to test what's going wrong
"""

import asyncio
import logging
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Your POINTS-Reader integration
from next_steps_guide import load_local_model, patch_pointsv15_for_mps, download_local_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOCRService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_loaded = False
        
    def initialize_model_sync(self):
        """Initialize model synchronously"""
        if self.model_loaded:
            return
            
        try:
            logger.info("=== STARTING MODEL INITIALIZATION ===")
            
            # Use your working setup
            model_dir = download_local_model(Path("POINTS-Reader-local"))
            patch_pointsv15_for_mps(model_dir)
            self.model, self.tokenizer, self.image_processor = load_local_model(Path("POINTS-Reader-local"))
            
            self.model_loaded = True
            logger.info("=== MODEL INITIALIZATION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"=== MODEL INITIALIZATION FAILED: {e} ===")
            raise

# Initialize service
ocr_service = SimpleOCRService()

# Create FastAPI app
app = FastAPI(title="Debug OCR App")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("=== STARTING DEBUG WEB APP ===")
    try:
        # Initialize model synchronously
        ocr_service.initialize_model_sync()
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple HTML page for testing"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Debug OCR App</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="p-8">
        <div class="max-w-2xl mx-auto">
            <h1 class="text-3xl font-bold mb-6">Debug OCR App</h1>
            
            <div class="bg-white shadow rounded-lg p-6 mb-6">
                <h2 class="text-lg font-semibold mb-4">Model Status</h2>
                <div id="status" class="mb-4">
                    <span class="inline-block w-3 h-3 rounded-full mr-2" 
                          style="background-color: {'green' if ocr_service.model_loaded else 'red'}"></span>
                    <span>{'Model Ready' if ocr_service.model_loaded else 'Model Not Ready'}</span>
                </div>
                <button onclick="checkStatus()" 
                        class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                    Refresh Status
                </button>
            </div>
            
            <div class="bg-white shadow rounded-lg p-6">
                <h2 class="text-lg font-semibold mb-4">Test File Upload</h2>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" name="file" accept=".png,.jpg,.jpeg,.pdf" 
                           class="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100">
                    <button type="submit" 
                            class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                        Test Upload
                    </button>
                </form>
                <div id="result" class="mt-4 p-4 bg-gray-100 rounded hidden"></div>
            </div>
        </div>

        <script>
            async function checkStatus() {{
                try {{
                    const response = await fetch('/health');
                    const data = await response.json();
                    console.log('Status check:', data);
                    location.reload(); // Simple refresh to update status
                }} catch (error) {{
                    console.error('Status check failed:', error);
                }}
            }}

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {{
                    alert('Please select a file first');
                    return;
                }}
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                resultDiv.innerHTML = 'Testing upload...';
                resultDiv.classList.remove('hidden');
                
                try {{
                    const response = await fetch('/test-upload', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    resultDiv.innerHTML = 'Upload test result: ' + JSON.stringify(data, null, 2);
                    console.log('Upload result:', data);
                }} catch (error) {{
                    resultDiv.innerHTML = 'Upload failed: ' + error.message;
                    console.error('Upload error:', error);
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Process uploaded file with OCR"""
    try:
        logger.info(f"=== PROCESSING FILE ===")
        logger.info(f"File: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        logger.info(f"Model loaded: {ocr_service.model_loaded}")
        
        if not ocr_service.model_loaded:
            return {"status": "error", "error": "Model not loaded"}
        
        # Save uploaded file temporarily
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        file_path = uploads_dir / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Process with OCR using your working functions
        from next_steps_guide import pdf_to_clean_pngs, prep_existing_png
        
        work_root = Path("work")
        pages_root = work_root / "pages"
        pages_root.mkdir(parents=True, exist_ok=True)
        
        stem = file_path.stem
        
        # Convert file to processed images
        if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            processed_images = prep_existing_png(file_path, pages_root, stem)
        else:
            processed_images = pdf_to_clean_pngs(file_path, pages_root, stem, dpi=300, max_pages=None)
        
        if not processed_images:
            return {"status": "error", "error": "No images could be processed from the file"}
        
        logger.info(f"Processed {len(processed_images)} images")
        
        # Run OCR on the first image
        first_image = processed_images[0]
        prompt = "Extract all text from this image. Return tables in HTML format and other text in Markdown format."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(first_image)},
                {"type": "text", "text": prompt}
            ]
        }]
        
        import torch
        with torch.inference_mode():
            result = ocr_service.model.chat(
                messages, 
                ocr_service.tokenizer, 
                ocr_service.image_processor, 
                {"max_new_tokens": 2048, "temperature": 0.0}
            )
        
        logger.info("OCR processing completed")
        
        # Clean up temporary files
        file_path.unlink()
        
        return {
            "status": "success",
            "filename": file.filename,
            "result": result,
            "pages_processed": len(processed_images)
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": ocr_service.model_loaded
    }

if __name__ == "__main__":
    print("=== STARTING DEBUG WEB APP ===")
    print("URL: http://localhost:8000")
    print("This will test model loading and file upload separately")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
