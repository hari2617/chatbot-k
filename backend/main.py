from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import base64
import os
from typing import Optional

# Import our modules
import chat
import pdf_rag
import image_qa


app = FastAPI(title="Groq Chatbot API")

# For Render free tier, skip heavy model loading at startup to avoid hanging
# Models will be loaded on first use instead
pass  # Placeholder to maintain structure

@app.get("/")
async def health_check():
    """Health check endpoint for Render warm-up"""
    return {"status": "healthy", "message": "Server is running"}

@app.get("/warmup")
async def warmup():
    """Warm-up endpoint to initialize models and resources"""
    # Trigger loading of embedding model if needed
    from pdf_rag import get_embedding_model
    try:
        model = get_embedding_model()  # This will load the model if not already loaded
        return {"status": "warmed_up", "message": "Models and resources initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class ChatRequest(BaseModel):
    prompt: str
    groq_api_key: str

class PDFQueryRequest(BaseModel):
    question: str
    groq_api_key: str
    pdf_content: str  # Base64 encoded PDF content for stateless processing

class ImageQueryRequest(BaseModel):
    image_base64: Optional[str] = None
    question: str
    openai_api_key: str

class ApiKeysRequest(BaseModel):
    groq_api_key: str
    openai_api_key: str

# --- Endpoints ---


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = chat.get_groq_chat_response(request.prompt, request.groq_api_key)
    if isinstance(response, dict) and "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return {"response": response}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        content = await file.read()
        # Process the PDF to verify it's valid
        result = pdf_rag.process_pdf(content)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        # Return success but don't store anything (stateless)
        return result  # Return the actual result from process_pdf
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-pdf")
async def ask_pdf(request: PDFQueryRequest):
    import base64
    try:
        # Decode the base64 PDF content
        pdf_bytes = base64.b64decode(request.pdf_content)
        # Process the PDF
        result = pdf_rag.process_pdf(pdf_bytes)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Now ask the question with the processed data
        answer = pdf_rag.ask_pdf_with_data(request.question, request.groq_api_key, result["vector_index"], result["chunks"])
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # This endpoint just converts uploaded image to base64 and returns it to frontend?
    # Or frontend can convert it. 
    # User flow: "User uploads an image ... Backend converts image to Base64"
    # User Request "Backend API Endpoints": POST /upload-image, POST /ask-image
    # This implies stateful image handling or returning an ID? 
    # The PROMPT says: "Backend Converts image to Base64". 
    # But then "User asks a question". 
    # If the backend is stateless (except for ApiKey and PDF Index per run), we should probably store the image in memory too?
    # "Backend stores the key in a global variable (prototype only)".
    # I will assume we store the LAST uploaded image in memory for simplicity for the "ask-image" endpoint.
    
    try:
        content = await file.read()
        encoded = base64.b64encode(content).decode('utf-8')
        print(f"Image uploaded successfully: {file.filename}, size: {len(content)} bytes")
        print(f"Base64 encoded size: {len(encoded)} characters")
        return {"message": "Image uploaded and processed", "image_base64": encoded}
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-image")
async def ask_image(request: ImageQueryRequest):
    # Image must be provided in the request
    img = request.image_base64
    
    if not img:
        raise HTTPException(status_code=400, detail="No image provided. Image must be sent with the request.")
    
    try:
        answer = image_qa.ask_image(img, request.question, request.openai_api_key)
        return {"answer": answer}
    except Exception as e:
        print(f"Error in ask_image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image question: {str(e)}")