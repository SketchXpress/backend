import os
import logging
import torch
import time
from datetime import timedelta
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init
from typing import Optional, List, Dict

from app.model.enhanced_model import EnhancedDiffusionModel
from app.model.error_handler import ModelErrorHandler

logger = logging.getLogger("celery_worker")
logging.basicConfig(level=logging.INFO)

# Define directories (absolute paths inside container)
UPLOAD_DIR = "/app/app/uploads"
GENERATED_DIR = "/app/app/generated"
CLEANUP_AGE_SECONDS = 3600 # 1 hour

# Configure Celery
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
    task_acks_late=True,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "cleanup-old-files-hourly": {
            "task": "app.worker.cleanup_old_files_task",
            "schedule": crontab(minute=0, hour='*'),
        },
    },
    timezone="UTC",
)

# Global variable to hold the model instance
model_instance: Optional[EnhancedDiffusionModel] = None

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize the model when a Celery worker process starts."""
    global model_instance
    logger.info("Initializing model for Celery worker...")
    try:
        use_cuda = torch.cuda.is_available()
        # Initialize with ONLY supported parameters
        model_instance = EnhancedDiffusionModel(
            output_dir=GENERATED_DIR,
            use_cuda=use_cuda
        )
        
        # Disable safety features AFTER initialization
        if hasattr(model_instance, 'pipe'):
            if hasattr(model_instance.pipe, 'safety_checker'):
                model_instance.pipe.safety_checker = None
                logger.warning("Safety checker has been forcibly removed.")
                
            if hasattr(model_instance.pipe, 'requires_safety_checker'):
                model_instance.pipe.requires_safety_checker = False
                
            if hasattr(model_instance.pipe, 'config') and hasattr(model_instance.pipe.config, 'requires_safety_checker'):
                model_instance.pipe.config.requires_safety_checker = False
        
        # Use custom methods if available
        if hasattr(model_instance, 'set_safety_checker_enabled'):
            model_instance.set_safety_checker_enabled(False)
            
        if hasattr(model_instance, 'enable_nsfw_generation'):
            model_instance.enable_nsfw_generation()
            
        logger.warning("⚠️ Safety checker has been disabled for image generation.")
        logger.info("Model initialized successfully in Celery worker.")
    except Exception as e:
        logger.error(f"Failed to initialize model in Celery worker: {e}", exc_info=True)

@celery_app.task
def cleanup_old_files_task():
    """Celery task to delete files older than CLEANUP_AGE_SECONDS."""
    now = time.time()
    deleted_count = 0
    logger.info(f"Running cleanup task for files older than {CLEANUP_AGE_SECONDS} seconds.")

    for directory in [UPLOAD_DIR, GENERATED_DIR]:
        logger.info(f"Checking directory: {directory}")
        try:
            if not os.path.isdir(directory):
                logger.warning(f"Directory not found: {directory}. Skipping cleanup.")
                continue

            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        file_mod_time = os.path.getmtime(file_path)
                        if (now - file_mod_time) > CLEANUP_AGE_SECONDS:
                            os.remove(file_path)
                            logger.info(f"Deleted old file: {file_path}")
                            deleted_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error listing directory {directory}: {e}")

    logger.info(f"Cleanup task finished. Deleted {deleted_count} files.")
    return {"status": "completed", "deleted_count": deleted_count}


@celery_app.task(bind=True)
def analyze_sketch_task(
    self,
    sketch_path: str,
    job_id: Optional[str] = None
) -> Dict:
    """Celery task to analyze a sketch using the pre-loaded model."""
    global model_instance
    if model_instance is None:
        logger.error("Model not initialized in worker. Cannot process task.")
        self.update_state(state="FAILURE", meta={"message": "Model not initialized"})
        raise RuntimeError("Model not initialized in worker")

    if job_id is None:
        job_id = self.request.id

    logger.info(f"Starting sketch analysis task {job_id}")

    try:
        analysis_result = model_instance.analyze_sketch(sketch_path)
        
        logger.info(f"Sketch analysis complete for task {job_id}")
        return {
            "status": "completed",
            "analysis": analysis_result
        }

    except Exception as e:
        logger.error(f"Error in sketch analysis task {job_id}: {str(e)}", exc_info=True)
        error_msg = ModelErrorHandler.handle_generation_error(e, job_id)
        self.update_state(state="FAILURE", meta={"message": error_msg})
        raise RuntimeError(f"Sketch analysis failed: {error_msg}")


@celery_app.task(bind=True)
def generate_images_task(
    self,
    sketch_path: str,
    prompt: Optional[str],
    negative_prompt: str,
    num_images: int,
    num_inference_steps: int,
    seed: Optional[int],
    temperature: float,
    guidance_scale: float,
    job_id: str
) -> Dict:
    """Celery task to generate images using the pre-loaded model."""
    global model_instance
    if model_instance is None:
        logger.error("Model not initialized in worker. Cannot process task.")
        self.update_state(state="FAILURE", meta={"message": "Model not initialized"})
        raise RuntimeError("Model not initialized in worker")

    logger.info(f"Starting image generation task {job_id}")

    # Ensure safety is disabled for this specific task
    if hasattr(model_instance, 'pipe') and hasattr(model_instance.pipe, 'safety_checker'):
        model_instance.pipe.safety_checker = None
        
    if hasattr(model_instance, 'enable_nsfw_generation'):
        model_instance.enable_nsfw_generation()

    def progress_callback(progress: float, images: List[str]):
        """Callback to update Celery task state with progress."""
        relative_image_paths = [path.replace("/app/app", "") for path in images]
        self.update_state(state="PROGRESS", meta={
            "progress": progress,
            "images": relative_image_paths,
            "safety_disabled": True
        })

    try:
        # Try both with and without allow_nsfw parameter
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
                callback=progress_callback,
                allow_nsfw=True
            )
        except TypeError:
            # Fall back if parameter not supported
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

        relative_image_paths = [path.replace("/app/app", "") for path in image_paths]

        logger.info(f"Image generation complete for task {job_id}")
        return {
            "status": "completed",
            "progress": 100.0,
            "images": relative_image_paths,
            "safety_disabled": True
        }

    except Exception as e:
        logger.error(f"Error in image generation task {job_id}: {str(e)}", exc_info=True)
        error_msg = ModelErrorHandler.handle_generation_error(e, job_id)
        self.update_state(state="FAILURE", meta={"message": error_msg})
        raise RuntimeError(f"Image generation failed: {error_msg}")
