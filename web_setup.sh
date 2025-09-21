# Navigate to your existing ocr directory
cd ~/ocr
source points_reader_env/bin/activate

# Install web framework dependencies
pip install fastapi uvicorn jinja2 python-multipart aiofiles
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib


# Create directories for web app
mkdir templates static uploads
touch templates/index.html


# Invoice OCR Web Application Setup Guide

## 🎯 What You're Building

A complete web application that combines:
- **Your working POINTS-Reader OCR** (local processing)
- **Google Drive integration** (read PDFs, save results)
- **Beautiful web interface** (upload files, manage processing)
- **MCP server capabilities** (for Claude integration)

## 📋 Architecture Overview

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Web Interface     │    │   FastAPI Server     │    │   POINTS-Reader     │
│   (Upload/Browse)   │◄──►│   (API + MCP)        │◄──►│   (Local OCR)       │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Google Drive API   │
                           │   (PDFs in/out)      │
                           └──────────────────────┘
```

## 🚀 Setup Instructions

### 1. Install Web Dependencies

```bash
# Navigate to your existing ocr directory
cd ~/ocr
source points_reader_env/bin/activate

# Install web framework dependencies
pip install fastapi uvicorn jinja2 python-multipart aiofiles
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 2. Create Project Structure

```bash
# Create directories for web app
mkdir templates static uploads
touch templates/index.html
```

### 3. Save the Files

Save these files in your `~/ocr/` directory:

1. **`web_app.py`** - Main FastAPI application (from the first artifact)
2. **`templates/index.html`** - Web interface template (from the second artifact)

### 4. Update Your requirements.txt

```txt
# Add to your existing requirements.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
jinja2>=3.1.0
python-multipart>=0.0.6
aiofiles>=23.0.0
google-api-python-client>=2.0.0
google-auth-httplib2>=0.1.0
google-auth-oauthlib>=0.5.0
pdf2image>=3.1.0
```

### 5. Google Drive Setup (Same as Before)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select project
3. Enable Google Drive API
4. Create OAuth 2.0 credentials
5. Download JSON file (you'll upload this via web interface)

### 6. Run the Web Application

```bash
# Start the web server
python web_app.py
```

**Access your app at:** http://localhost:8000

## 🎨 Web Interface Features

### **Upload Tab**
- ✅ Drag & drop file upload
- ✅ Support for PNG, JPG, PDF files
- ✅ Custom OCR prompts
- ✅ Real-time processing progress

### **Google Drive Tab**
- ✅ Browse your PDF files
- ✅ One-click processing
- ✅ Auto-save results back to Drive

### **Setup Tab**
- ✅ Easy Google Drive connection
- ✅ Step-by-step instructions
- ✅ Credential file upload

## 🔧 Advanced Configuration

### Custom Deployment Options

#### Option 1: Local Network Access
```bash
# Allow access from other devices on your network
python -c "
import uvicorn
from web_app import app
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

#### Option 2: Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn web_app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

#### Option 3: Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
# Optional configuration
export OCR_MODEL_PATH="./POINTS-Reader-local"
export UPLOAD_MAX_SIZE="50MB"
export GOOGLE_DRIVE_FOLDER="Invoice_OCR_Results"
```

## 🔗 MCP Integration

Your web app **also functions as an MCP server**! You can connect it to Claude:

### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "invoice-ocr-web": {
      "command": "python",
      "args": ["/path/to/your/web_app.py", "--mcp-mode"],
      "env": {
        "PYTHONPATH": "/path/to/your/ocr"
      }
    }
  }
}
```

### Available MCP Tools
- `process_uploaded_image` - Process image via web interface
- `list_drive_pdfs` - Browse Google Drive PDFs
- `process_drive_pdf` - Process PDF from Google Drive
- `get_processing_status` - Check job status

## 📱 Usage Examples

### 1. Single File Processing
1. Go to **Upload tab**
2. Drag & drop an invoice PDF
3. Optionally add custom prompt: *"Extract invoice number, date, total amount as JSON"*
4. Click **Process Document**
5. Watch real-time progress
6. Copy/download results

### 2. Batch Processing from Google Drive
1. Go to **Setup tab** → Upload Google Drive credentials
2. Go to **Google Drive tab**
3. Click **Refresh** to see your PDFs
4. Click **Process** on any file
5. Results automatically saved to "OCR_Results" folder

### 3. Custom OCR Prompts
```
Extract the following fields as JSON:
{
  "invoice_number": "",
  "date": "",
  "vendor": "",
  "total_amount": "",
  "line_items": []
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Model not loading**
   ```bash
   # Check if POINTS-Reader is working
   python next_steps_guide.py --image test.png
   ```

2. **Google Drive connection fails**
   - Ensure OAuth 2.0 credentials are for "Desktop application"
   - Check JSON file format
   - Verify Google Drive API is enabled

3. **Web interface not accessible**
   ```bash
   # Check if port 8000 is free
   lsof -i :8000
   
   # Try different port
   uvicorn web_app:app --port 8080
   ```

4. **File upload fails**
   - Check file size (default limit: 50MB)
   - Ensure uploads/ directory exists
   - Verify file permissions

### Performance Optimization

1. **Reduce memory usage**
   ```python
   # In web_app.py, add model quantization
   model_kwargs = {"load_in_8bit": True}
   ```

2. **Faster processing**
   ```python
   # Process multiple files in parallel
   max_concurrent_jobs = 2
   ```

3. **Caching**
   ```python
   # Cache processed results
   enable_result_cache = True
   ```

## 🎉 Success!

You now have a **complete invoice processing solution**:

- ✅ **Beautiful web interface**
- ✅ **Local AI processing** (privacy + speed)
- ✅ **Google Drive integration**
- ✅ **MCP server capabilities**
- ✅ **Real-time progress tracking**
- ✅ **Batch processing support**

**Next Steps:**
1. Test with your actual invoices
2. Customize the interface styling
3. Add more OCR prompt templates
4. Set up automated processing workflows
5. Deploy to production if needed

Your invoice OCR web app is ready to use! 🚀
