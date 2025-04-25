'''
Main FastAPI server with /generate, /status/{job_id}, and /analyze endpoints.
'''
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid

from app.model.model_integration import ModelIntegration
from app.model.error_handler import ModelErrorHandler

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images publicly
if not os.path.exists("app/generated"):
    os.makedirs("app/generated")

app.mount("/generated", StaticFiles(directory="app/generated"), name="generated")

# Initialize AI model
model = ModelIntegration(output_dir="app/generated", upload_dir="app/uploads")


@app.post("/api/generate")
async def generate(
    sketch: UploadFile = File(...),
    prompt: str = Form(None),
    temperature: float = Form(0.65),
    guidance_scale: float = Form(7.5),
    num_images: int = Form(1),
    steps: int = Form(30),
    seed: int = Form(None)
):
    try:
        # Save the uploaded sketch to uploads/
        job_id = str(uuid.uuid4())
        sketch_path = f"app/uploads/{job_id}_{sketch.filename}"
        with open(sketch_path, "wb") as buffer:
            shutil.copyfileobj(sketch.file, buffer)

        # Start generation in the background
        model.generate_images(
            sketch_path=sketch_path,
            prompt=prompt,
            num_images=num_images,
            num_inference_steps=steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            seed=seed,
            job_id=job_id
        )

        return {"status": "started", "job_id": job_id}

    except Exception as e:
        error_msg = ModelErrorHandler.handle_generation_error(e, job_id=None)
        return ModelErrorHandler.create_fallback_response("generation_error", error_msg)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    return model.get_job_status(job_id)


@app.post("/api/analyze")
async def analyze(sketch: UploadFile = File(...)):
    try:
        path = f"app/uploads/analyze_{uuid.uuid4()}_{sketch.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(sketch.file, buffer)

        analysis = model.analyze_sketch(path)
        return {"status": "ok", "result": analysis}
    except Exception as e:
        error_msg = ModelErrorHandler.handle_generation_error(e, job_id=None)
        return ModelErrorHandler.create_fallback_response("analyze_error", error_msg)
