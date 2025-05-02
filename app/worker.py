# app/worker.py
import os
import logging
import torch
from celery import Celery
from celery.signals import worker_process_init
from typing import Optional, List, Dict

from app.model.enhanced_model import EnhancedDiffusionModel
from app.model.error_handler import ModelErrorHandler

logger = logging.getLogger("celery_worker")
logging.basicConfig(level=logging.INFO)

# Configure Celery
# Use Redis as the broker and result backend
# The REDIS_URL should be set as an environment variable, e.g., redis://redis:6379/0
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Ensure tasks are acknowledged only after completion/failure
    task_acks_late=True,
    # Set worker concurrency to 1 to avoid multiple models loading on one GPU
    worker_concurrency=1,
    # Prefetch multiplier 1 ensures worker only reserves one task at a time
    worker_prefetch_multiplier=1,
)

# Global variable to hold the model instance within the worker process
model_instance: Optional[EnhancedDiffusionModel] = None

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize the model when a Celery worker process starts."""
    global model_instance
    logger.info("Initializing model for Celery worker...")
    try:
        use_cuda = torch.cuda.is_available()
        model_instance = EnhancedDiffusionModel(
            output_dir="/app/app/generated", # Use absolute path inside container
            use_cuda=use_cuda
        )
        logger.info("Model initialized successfully in Celery worker.")
    except Exception as e:
        logger.error(f"Failed to initialize model in Celery worker: {e}", exc_info=True)
        # If model fails to load, the worker might not be able to process tasks.
        # Depending on desired behavior, could raise an exception to stop the worker.


@celery_app.task(bind=True)
def generate_images_task(
    self, # Celery task instance
    sketch_path: str,
    prompt: Optional[str],
    negative_prompt: str,
    num_images: int,
    num_inference_steps: int,
    seed: Optional[int],
    temperature: float,
    guidance_scale: float,
    job_id: str # Use the task ID as the job ID
) -> Dict:
    """Celery task to generate images using the pre-loaded model."""
    global model_instance
    if model_instance is None:
        logger.error("Model not initialized in worker. Cannot process task.")
        # Update task state to failure
        self.update_state(state="FAILURE", meta={"message": "Model not initialized"})
        # Use ignore_result=True or raise Ignore() if results aren't needed
        # Or raise an exception to signify failure
        raise RuntimeError("Model not initialized in worker")

    logger.info(f"Starting image generation task {job_id}")

    def progress_callback(progress: float, images: List[str]):
        """Callback to update Celery task state with progress."""
        # Map image paths to be relative URLs for the API response
        relative_image_paths = [path.replace("/app/app", "") for path in images]
        self.update_state(state="PROGRESS", meta={
            "progress": progress,
            "images": relative_image_paths
        })

    try:
        image_paths = model_instance.generate_images_with_progress(
            sketch_path=sketch_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            seed=seed,
            temperature=temperature,
            guidance_scale=guidance_scale,
            job_id=job_id,
            callback=progress_callback
        )

        # Map final image paths to relative URLs
        relative_image_paths = [path.replace("/app/app", "") for path in image_paths]

        logger.info(f"Image generation complete for task {job_id}")
        # Return final result (Celery backend stores this)
        return {
            "status": "completed",
            "progress": 100.0,
            "images": relative_image_paths
        }

    except Exception as e:
        logger.error(f"Error in image generation task {job_id}: {str(e)}", exc_info=True)
        error_msg = ModelErrorHandler.handle_generation_error(e, job_id)
        # Update task state to failure with error message
        self.update_state(state="FAILURE", meta={"message": error_msg})
        # Re-raise the exception so Celery knows the task failed
        raise

