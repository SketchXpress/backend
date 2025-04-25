'''
Core sketch-to-image AI engine using ControlNet and Stable Diffusion.
'''
import os
import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Callable, Any, Union, Dict, cast
import uuid
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    DiffusionPipeline
)
from diffusers.schedulers import DDPMScheduler
from transformers import pipeline

logger = logging.getLogger("enhanced_model")
logging.basicConfig(level=logging.INFO)

class EnhancedDiffusionModel:
    def __init__(
        self,
        temperature: float = 0.65,
        guidance_scale: float = 7.5,
        batch_size: int = 4,
        output_dir: str = "app/generated",
        use_cuda: bool = True,
        controlnet_model: str = "lllyasviel/control_v11p_sd15_scribble",
        base_model: str = "runwayml/stable-diffusion-v1-5"
    ):
        self.temperature = temperature
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.controlnet_model = controlnet_model
        self.base_model = base_model
        self.pipe: Any = None  # Will be initialized as StableDiffusionControlNetPipeline
        self.controlnet: Any = None
        self.sketch_classifier: Any = None

        os.makedirs(output_dir, exist_ok=True)

        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info("Initializing model...")

            torch_dtype = torch.float16 if self.use_cuda else torch.float32

            # Load ControlNet model
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch_dtype
            )

            # Load base Stable Diffusion pipeline with ControlNet
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model,
                controlnet=self.controlnet,
                torch_dtype=torch_dtype
            )

            # Use correct type for scheduler replacement
            # We use the from_config method which returns a properly typed scheduler
            new_scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            # Type assertion to help Pylance understand
            self.pipe.scheduler = cast(DDPMScheduler, new_scheduler)

            if self.use_cuda:
                self.pipe.enable_model_cpu_offload()

            logger.info("Pipeline and scheduler initialized successfully")

            # Load sketch analysis model
            self._initialize_sketch_recognition()

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _initialize_sketch_recognition(self):
        try:
            self.sketch_classifier = pipeline("image-classification", model="microsoft/resnet-50")
        except Exception as e:
            logger.error(f"Sketch recognition model failed to load: {e}")
            raise

    def analyze_sketch(self, sketch_path: str) -> dict:
        try:
            sketch = Image.open(sketch_path).convert("RGB")
            results = self.sketch_classifier(sketch)
            top = list(results)[:3] if results else []

            return {
                "top_class": top[0].get('label', "unknown") if isinstance(top[0], dict) else "unknown",
                "top_score": top[0].get('score', 0.0) if isinstance(top[0], dict) else 0.0,
                "predictions": top,
                "sketch_size": sketch.size
            }
        except Exception as e:
            logger.error(f"Sketch analysis failed: {e}")
            raise

    def generate_images_with_progress(
        self,
        sketch_path: str,
        prompt: Optional[str] = None,
        negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
        num_images: int = 4,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        save_results: bool = True,
        temperature: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        job_id: Optional[str] = None,
        callback: Optional[Callable[[float, List[str]], None]] = None,
        styles: Optional[List[str]] = None
    ) -> List[str]:
        if job_id is None:
            job_id = str(uuid.uuid4())

        logger.info(f"Starting generation for job {job_id}")

        if temperature is not None:
            self.temperature = temperature
        if guidance_scale is not None:
            self.guidance_scale = guidance_scale

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            generator = None

        try:
            sketch = Image.open(sketch_path).convert("RGB")
            sketch = sketch.resize((512, 512))

            if prompt is None:
                analysis = self.analyze_sketch(sketch_path)
                prompt = f"high quality, detailed {analysis['top_class']}, professional"

            logger.info(f"Using prompt: {prompt}")
        except Exception as e:
            logger.error(f"Failed to load sketch: {e}")
            if callback:
                callback(0.0, [])
            raise

        if styles is None:
            styles = [
                "realistic, detailed, high resolution",
                "artistic, colorful, creative",
                "minimalist, clean lines, simple",
                "professional, technical illustration"
            ]

        image_paths = []
        total_steps = num_images * num_inference_steps
        current_step = 0

        # Ensure pipe is properly typed as a callable object
        pipe = cast(StableDiffusionControlNetPipeline, self.pipe)

        for i in range(num_images):
            try:
                styled_prompt = f"{prompt}, {styles[i % len(styles)]}"

                # Use the pipe with correct typing
                output = pipe(
                    prompt=styled_prompt,
                    image=sketch,  # This is correct for ControlNetPipeline
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    return_dict=True
                )

                images = getattr(output, "images", None)
                if images and isinstance(images, list):
                    img_path = os.path.join(self.output_dir, f"{job_id}_{i}.png")
                    images[0].save(img_path)
                    image_paths.append(img_path)

                current_step += num_inference_steps
                if callback:
                    callback((current_step / total_steps) * 100.0, image_paths)

            except Exception as e:
                logger.error(f"Error generating image {i+1}: {e}")
                continue

        if callback:
            callback(100.0, image_paths)

        logger.info(f"Completed {len(image_paths)} images for job {job_id}")
        return image_paths

    def rank_images(self, image_paths: List[str]) -> List[str]:
        return image_paths