#!/usr/bin/env python3
"""
Web-Specific POINTS-Reader Wrapper
Provides isolated processing for the web app while using your existing next_steps_guide.py functions
"""

import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import uuid

# Import your working functions
from next_steps_guide import (
    load_local_model, patch_pointsv15_for_mps, download_local_model,
    pdf_to_clean_pngs, prep_existing_png, run_chat_on_images
)

logger = logging.getLogger(__name__)

class WebOCRWrapper:
    """
    Web-specific wrapper that provides isolated processing using your CLI functions
    Each job gets a unique work directory to prevent conflicts
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_loaded = False
        
        # Create a persistent base work directory for the web app
        self.base_work_dir = Path("web_work")
        self.base_work_dir.mkdir(exist_ok=True)
        
        # Initialize model on startup
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model once on startup"""
        try:
            logger.info("=== INITIALIZING POINTS-READER MODEL FOR WEB ===")
            
            # Use your existing functions
            model_dir = download_local_model(Path("POINTS-Reader-local"))
            patch_pointsv15_for_mps(model_dir)
            self.model, self.tokenizer, self.image_processor = load_local_model(Path("POINTS-Reader-local"))
            
            self.model_loaded = True
            logger.info("=== WEB MODEL INITIALIZATION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def process_single_file(self, file_path: Path, custom_prompt: str = None, job_id: str = None) -> Dict:
        """
        Process a single file with isolated work directory and automatic saving
        
        Args:
            file_path: Path to the uploaded file
            custom_prompt: Custom OCR prompt
            job_id: Unique job identifier for isolation
            
        Returns:
            Dict with processing results and metadata
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        # Create isolated work directories for this job
        job_work_dir = self.base_work_dir / f"job_{job_id}"
        job_pages_dir = job_work_dir / "pages"
        job_output_dir = job_work_dir / "output"  # Add output directory
        job_pages_dir.mkdir(parents=True, exist_ok=True)
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use job_id as unique stem to prevent conflicts
        # Note: file_path already contains job_id prefix from web app
        original_stem = file_path.stem.split('_', 1)[-1] if '_' in file_path.stem else file_path.stem
        unique_stem = f"{job_id}_{original_stem}"
        
        try:
            logger.info(f"Processing {file_path.name} with unique stem: {unique_stem}")
            
            # Step 1: Convert to processed images using your functions
            if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                processed_images = prep_existing_png(file_path, job_pages_dir, unique_stem)
            else:  # PDF
                processed_images = pdf_to_clean_pngs(
                    file_path, job_pages_dir, unique_stem, 
                    dpi=300, max_pages=None
                )
            
            if not processed_images:
                raise RuntimeError("No images could be processed from the file")
            
            logger.info(f"Created {len(processed_images)} processed images")
            
            # Step 2: Run OCR using your exact function
            prompt = custom_prompt or "Extract all text. Tables as HTML; other text as Markdown."
            
            page_texts, combined_md = run_chat_on_images(
                self.model, self.tokenizer, self.image_processor,
                processed_images, prompt,
                max_new_tokens=2048, debug=False
            )
            
            logger.info("OCR processing completed successfully")
            
            # Step 3: Save outputs like next_steps_guide.py does
            self._save_outputs(unique_stem, page_texts, combined_md, job_output_dir, file_path.name)
            
            # Step 4: Cleanup job work directory after a delay (keep outputs briefly)
            # Don't cleanup immediately - keep outputs for download/reference
            
            return {
                "status": "success",
                "result": combined_md,
                "pages_processed": len(processed_images),
                "page_texts": page_texts,
                "original_filename": file_path.name,
                "output_directory": str(job_output_dir),
                "saved_files": {
                    "combined_md": str(job_output_dir / f"{unique_stem}.md"),
                    "page_files": [str(job_output_dir / f"{unique_stem}_page_{i+1:03d}.txt") for i in range(len(page_texts))]
                }
            }
            
        except Exception as e:
            logger.error(f"Processing failed for job {job_id}: {e}")
            # Cleanup on error
            self._cleanup_job_directory(job_work_dir)
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def _save_outputs(self, stem: str, page_texts: List[str], combined_md: str, output_dir: Path, original_filename: str):
        """Save outputs like next_steps_guide.py does"""
        try:
            # Save per-page text files
            for i, txt in enumerate(page_texts, start=1):
                page_file = output_dir / f"{stem}_page_{i:03d}.txt"
                page_file.write_text(txt if txt else "", encoding="utf-8")
            
            # Save combined markdown
            combo_file = output_dir / f"{stem}.md"
            combo_file.write_text(combined_md if combined_md else "", encoding="utf-8")
            
            # Save a simple manifest for reference
            manifest = {
                "original_filename": original_filename,
                "stem": stem,
                "pages_processed": len(page_texts),
                "processed_at": str(uuid.uuid4()),  # Use timestamp alternative
                "files": {
                    "combined": f"{stem}.md",
                    "pages": [f"{stem}_page_{i:03d}.txt" for i in range(1, len(page_texts) + 1)]
                }
            }
            
            manifest_file = output_dir / "manifest.json"
            manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            
            logger.info(f"Saved outputs to: {output_dir}")
            logger.info(f"Files: {combo_file}, {len(page_texts)} page files, manifest.json")
            
        except Exception as e:
            logger.error(f"Failed to save outputs: {e}")
            raise
    
    def _cleanup_job_directory(self, job_dir: Path):
        """Clean up isolated job directory after processing"""
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
                logger.info(f"Cleaned up job directory: {job_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {job_dir}: {e}")
    
    def get_model_info(self) -> Dict:
        """Get model status information"""
        return {
            "model_loaded": self.model_loaded,
            "device": next(self.model.parameters()).device.type if self.model_loaded else "unknown",
            "work_directory": str(self.base_work_dir)
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old job directories (maintenance function)"""
        import time
        current_time = time.time()
        cleanup_count = 0
        
        try:
            for job_dir in self.base_work_dir.glob("job_*"):
                if job_dir.is_dir():
                    # Check directory age
                    dir_age_hours = (current_time - job_dir.stat().st_mtime) / 3600
                    if dir_age_hours > max_age_hours:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        cleanup_count += 1
                        
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old job directories")
                
        except Exception as e:
            logger.warning(f"Job cleanup failed: {e}")

# Create global instance
web_ocr = WebOCRWrapper()
