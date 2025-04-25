'''
Connects the API to the model, handles background jobs and progress tracking.
'''
import os
import logging
import uuid
import threading
import torch
from typing import Optional, List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor

from app.model.enhanced_model import EnhancedDiffusionModel
from app.model.error_handler import ModelErrorHandler

logger = logging.getLogger("model_integration")
logging.basicConfig(level=logging.INFO)

class ModelIntegration:
    """
    Integration class to connect the enhanced diffusion model with the web application
    """

    def __init__(
        self,
        output_dir: str = "app/generated",
        upload_dir: str = "app/uploads",
        use_cuda: bool = True
    ):
        self.output_dir = output_dir
        self.upload_dir = upload_dir
        self.use_cuda = use_cuda and torch.cuda.is_available()

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)

        self.jobs: Dict[str, Dict] = {}
        self.job_lock = threading.Lock()

        self.executor = ThreadPoolExecutor(max_workers=2)

        self.model = EnhancedDiffusionModel(
            output_dir=output_dir,
            use_cuda=self.use_cuda
        )

        logger.info("Model integration initialized successfully")

    def generate_images(
        self,
        sketch_path: str,
        prompt: Optional[str],
        negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
        num_images: int = 4,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        temperature: float = 0.65,
        guidance_scale: float = 7.5,
        job_id: Optional[str] = None
    ) -> str:
        if job_id is None:
            job_id = str(uuid.uuid4())

        logger.info(f"Starting image generation job {job_id}")

        with self.job_lock:
            self.jobs[job_id] = {
                "status": "processing",
                "progress": 0.0,
                "images": [],
                "message": None
            }

        def progress_callback(progress: float, images: List[str]):
            with self.job_lock:
                self.jobs[job_id]["progress"] = progress
                self.jobs[job_id]["images"] = images

        self.executor.submit(
            self._generate_images_thread,
            job_id,
            sketch_path,
            prompt or "",
            negative_prompt,
            num_images,
            num_inference_steps,
            seed,
            temperature,
            guidance_scale,
            progress_callback
        )

        return job_id

    def _generate_images_thread(
        self,
        job_id: str,
        sketch_path: str,
        prompt: str,
        negative_prompt: str,
        num_images: int,
        num_inference_steps: int,
        seed: Optional[int],
        temperature: float,
        guidance_scale: float,
        callback: Callable[[float, List[str]], None]
    ):
        try:
            logger.info(f"Thread started for job {job_id}")

            image_paths = self.model.generate_images_with_progress(
                sketch_path=sketch_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=num_images,
                num_inference_steps=num_inference_steps,
                seed=seed,
                temperature=temperature,
                guidance_scale=guidance_scale,
                job_id=job_id,
                callback=callback
            )

            with self.job_lock:
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["progress"] = 100.0
                self.jobs[job_id]["images"] = image_paths

            logger.info(f"Image generation complete for job {job_id}")

        except Exception as e:
            logger.error(f"Error in job {job_id}: {str(e)}")
            with self.job_lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["message"] = ModelErrorHandler.handle_generation_error(e, job_id)

    def get_job_status(self, job_id: str) -> Dict:
        with self.job_lock:
            if job_id in self.jobs:
                return self.jobs[job_id].copy()
            else:
                return {
                    "status": "not_found",
                    "progress": 0.0,
                    "images": [],
                    "message": "Job not found"
                }

    def analyze_sketch(self, sketch_path: str) -> Dict:
        return self.model.analyze_sketch(sketch_path)
