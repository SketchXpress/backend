# app/api.py
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult

# Import the Celery app instance and the task
from app.worker import celery_app, generate_images_task
# We might need a separate task for analysis if we want it async
# from app.worker import analyze_sketch_task # Assuming this exists

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "https://api.sketchxpress.tech", "https://sketchxpress.tech"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define directories (relative to WORKDIR /app)
UPLOAD_DIR = "app/uploads"
GENERATED_DIR = "app/generated"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# Serve generated images publicly (relative to WORKDIR /app)
# The path in StaticFiles must match the container path
# The mount path "/generated" is the URL path
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

# No need to initialize ModelIntegration here anymore

@app.post("/api/generate")
async def generate(
    sketch: UploadFile = File(...),
    prompt: str = Form(None),
    temperature: float = Form(0.65),
    guidance_scale: float = Form(7.5),
    num_images: int = Form(1), # Default back to 1 for faster testing?
    steps: int = Form(30),
    seed: int = Form(None)
):
    try:
        # Create a unique job ID (can use UUID)
        job_id = str(uuid.uuid4())
        # Save the uploaded sketch using an absolute path within the container
        sketch_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{job_id}_{sketch.filename}"))
        with open(sketch_path, "wb") as buffer:
            shutil.copyfileobj(sketch.file, buffer)

        # Send the task to the Celery queue
        task_result = generate_images_task.apply_async(
            args=[
                sketch_path,
                prompt,
                "lowres, bad anatomy, bad hands, cropped, worst quality", # negative_prompt
                num_images,
                steps,
                seed,
                temperature,
                guidance_scale,
                job_id # Pass job_id to task
            ],
            task_id=job_id # Use our generated UUID as the Celery task ID
        )

        return {"status": "queued", "job_id": task_result.id}

    except Exception as e:
        # Basic error handling for API level issues (e.g., file save)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to queue task: {str(e)}"}
        )

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a Celery task."""
    task_result = AsyncResult(job_id, app=celery_app)

    response = {
        "job_id": job_id,
        "status": task_result.state, # PENDING, STARTED, RETRY, FAILURE, SUCCESS, PROGRESS
        "progress": 0.0,
        "images": [],
        "message": None
    }

    if task_result.state == "PENDING":
        response["message"] = "Task is waiting in the queue."
    elif task_result.state == "STARTED":
        response["message"] = "Task has started processing."
    elif task_result.state == "PROGRESS":
        response["progress"] = task_result.info.get("progress", 0.0)
        response["images"] = task_result.info.get("images", [])
        response["message"] = "Task is in progress."
    elif task_result.state == "SUCCESS":
        result = task_result.get() # Get the final result dict
        response["status"] = result.get("status", "completed") # Should be 'completed'
        response["progress"] = result.get("progress", 100.0)
        response["images"] = result.get("images", [])
        response["message"] = "Task completed successfully."
    elif task_result.state == "FAILURE":
        response["message"] = str(task_result.info) # Get the exception info
        # Optionally, retrieve custom error message if set in task failure meta
        if isinstance(task_result.info, dict):
             response["message"] = task_result.info.get("message", str(task_result.info))

    # Handle case where job_id is not found (state will be PENDING, but maybe add explicit check?)
    # Celery doesn't easily distinguish 'not found' from 'pending' without backend query

    return response

# TODO: Re-implement /api/analyze if needed, potentially as another Celery task
# @app.post("/api/analyze")
# async def analyze(sketch: UploadFile = File(...)):
#     # ... save file ...
#     # task_result = analyze_sketch_task.delay(sketch_path)
#     # return {"status": "queued", "job_id": task_result.id}
#     pass

