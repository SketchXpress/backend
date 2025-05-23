# app/api.py
import os
import uuid
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import time

# Import from optimized worker instead of the original worker
from app.optimized_worker import celery_app, generate_images_task, analyze_sketch_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Create FastAPI app
app = FastAPI(
    title="SketchXpress API",
    description="API for transforming sketches into detailed images using AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
UPLOAD_DIR = "app/uploads"
GENERATED_DIR = "app/generated"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# Mount static directories
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

# Define response models
class GenerationResponse(BaseModel):
    status: str
    job_id: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    images: List[str] = []
    message: str = ""

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    analysis: Dict[str, Any] = {}
    message: str = ""

@app.get("/")
async def root():
    return {"message": "SketchXpress API is running. Visit /docs for documentation."}

@app.post("/api/generate", response_model=GenerationResponse)
async def generate_images(
    sketch: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    negative_prompt: str = Form("lowres, bad anatomy, bad hands, cropped, worst quality"),
    temperature: float = Form(0.65),
    guidance_scale: float = Form(7.5),
    num_images: int = Form(1),
    steps: int = Form(30),
    seed: Optional[int] = Form(None)
):
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save the uploaded sketch
        sketch_filename = f"{job_id}_sketch.png"
        sketch_path = os.path.join(UPLOAD_DIR, sketch_filename)
        
        with open(sketch_path, "wb") as f:
            f.write(await sketch.read())
        
        logger.info(f"Saved sketch to {sketch_path}")
        
        # Submit the generation task to Celery
        task = generate_images_task.delay(
            sketch_path=sketch_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_inference_steps=steps,
            seed=seed,
            temperature=temperature,
            guidance_scale=guidance_scale,
            job_id=job_id
        )
        
        logger.info(f"Submitted generation task with ID: {job_id}")
        
        return {"status": "queued", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Error in generate_images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_sketch(
    sketch: UploadFile = File(...),
):
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save the uploaded sketch
        sketch_filename = f"{job_id}_sketch.png"
        sketch_path = os.path.join(UPLOAD_DIR, sketch_filename)
        
        with open(sketch_path, "wb") as f:
            f.write(await sketch.read())
        
        logger.info(f"Saved sketch to {sketch_path}")
        
        # Submit the analysis task to Celery
        task = analyze_sketch_task.delay(
            sketch_path=sketch_path,
            job_id=job_id
        )
        
        logger.info(f"Submitted analysis task with ID: {job_id}")
        
        return {"status": "queued", "job_id": job_id, "analysis": {}, "message": "Analysis queued"}
        
    except Exception as e:
        logger.error(f"Error in analyze_sketch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def check_status(job_id: str):
    try:
        # Check if the task exists
        task = celery_app.AsyncResult(job_id)
        
        if task.state == "PENDING":
            return {
                "job_id": job_id,
                "status": "pending",
                "progress": 0.0,
                "images": [],
                "message": "Task is pending."
            }
        elif task.state == "PROGRESS":
            # For tasks that report progress
            meta = task.info or {}
            return {
                "job_id": job_id,
                "status": "in_progress",
                "progress": meta.get("progress", 0.0),
                "images": meta.get("images", []),
                "message": "Task is in progress."
            }
        elif task.state == "SUCCESS":
            result = task.result
            return {
                "job_id": job_id,
                "status": "completed",
                "progress": 100.0,
                "images": result.get("images", []),
                "message": "Task completed successfully."
            }
        elif task.state == "FAILURE":
            return {
                "job_id": job_id,
                "status": "failed",
                "progress": 0.0,
                "images": [],
                "message": str(task.info)
            }
        else:
            return {
                "job_id": job_id,
                "status": task.state.lower(),
                "progress": 0.0,
                "images": [],
                "message": f"Task is in state: {task.state}"
            }
            
    except Exception as e:
        logger.error(f"Error in check_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{job_id}", response_model=AnalysisResponse)
async def check_analysis(job_id: str):
    try:
        # Check if the task exists
        task = celery_app.AsyncResult(job_id)
        
        if task.state == "PENDING":
            return {
                "job_id": job_id,
                "status": "pending",
                "analysis": {},
                "message": "Analysis is pending."
            }
        elif task.state == "SUCCESS":
            result = task.result
            return {
                "job_id": job_id,
                "status": "completed",
                "analysis": result.get("analysis", {}),
                "message": "Analysis completed successfully."
            }
        elif task.state == "FAILURE":
            return {
                "job_id": job_id,
                "status": "failed",
                "analysis": {},
                "message": str(task.info)
            }
        else:
            return {
                "job_id": job_id,
                "status": task.state.lower(),
                "analysis": {},
                "message": f"Analysis is in state: {task.state}"
            }
            
    except Exception as e:
        logger.error(f"Error in check_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
