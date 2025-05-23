import os
import logging
import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import List, Optional, Callable, Any, Union, Dict, cast, Tuple
import uuid
import importlib.util

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    AutoencoderKL,
    DiffusionPipeline
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, pipeline, CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torchvision.transforms import transforms

logger = logging.getLogger("optimized_model")
logging.basicConfig(level=logging.INFO)

# Global progress handler for diffusers pipelines
class DiffusersProgressCallback:
    """Global progress handler for diffusers pipelines"""
    active_callbacks = {}  # Map job_ids to callback functions
    active_progress = {}   # Track progress for each job
    active_images = {}     # Track images for each job

    @classmethod
    def register(cls, job_id, callback):
        """Register a callback for a job"""
        cls.active_callbacks[job_id] = callback
        cls.active_progress[job_id] = 0.0
        cls.active_images[job_id] = []

    @classmethod
    def update(cls, job_id, progress, images=None):
        """Update progress for a job"""
        if job_id in cls.active_callbacks:
            cls.active_progress[job_id] = progress
            if images is not None:
                cls.active_images[job_id] = images
            cls.active_callbacks[job_id](progress, cls.active_images[job_id])

    @classmethod
    def unregister(cls, job_id):
        """Clean up when job is done"""
        if job_id in cls.active_callbacks:
            del cls.active_callbacks[job_id]
        if job_id in cls.active_progress:
            del cls.active_progress[job_id]
        if job_id in cls.active_images:
            del cls.active_images[job_id]

# Global tqdm subclass that integrates with our progress system
from tqdm.auto import tqdm
class GlobalProgressTqdm(tqdm):
    """A tqdm subclass that reports to our global registry"""
    
    def __init__(self, *args, **kwargs):
        # Extract and remove our custom kwargs
        self.job_id = kwargs.pop("job_id", None)
        self.image_idx = kwargs.pop("image_idx", 0)
        self.total_images = kwargs.pop("total_images", 1)
        
        # Ensure the 'disable' attribute is set before super().__init__
        # This prevents AttributeError during cleanup in tqdm.__del__
        self.disable = kwargs.get('disable', False)
        
        # Handle tensor arguments safely
        # Convert any tensor arguments to Python scalars to avoid ambiguity errors
        for key, value in list(kwargs.items()):
            if hasattr(value, 'item') and callable(getattr(value, 'item')):
                try:
                    kwargs[key] = value.item()
                except:
                    # If conversion fails, use a safe default
                    if key == 'total':
                        kwargs[key] = 100
                    elif key == 'initial':
                        kwargs[key] = 0
        
        # Ensure total is a valid number
        if 'total' in kwargs and kwargs['total'] is None:
            kwargs['total'] = 100
            
        # Initialize all standard tqdm attributes to prevent AttributeError during cleanup
        self.n = 0
        self.total = kwargs.get('total', 100)
        self.desc = kwargs.get('desc', '')
        
        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error initializing tqdm: {e}")
            # Ensure all critical attributes exist even if initialization fails
            if not hasattr(self, 'n'):
                self.n = 0
            if not hasattr(self, 'total'):
                self.total = 100
        
    def update(self, n=1):
        """Override update to report progress to our global registry"""
        try:
            # Convert tensor n to scalar if needed
            if hasattr(n, 'item') and callable(getattr(n, 'item')):
                try:
                    n = n.item()
                except:
                    n = 1
            
            super().update(n)
            
            # Only report if we have a job_id
            if self.job_id and hasattr(self, 'total') and self.total:
                # Calculate overall progress as percentage
                progress = min(100.0, (
                    (self.image_idx * 100.0 / self.total_images) + 
                    (self.n * 100.0 / (self.total * self.total_images))
                ))
                
                # Report to the global registry
                DiffusersProgressCallback.update(self.job_id, progress)
        except Exception as e:
            logger.warning(f"Error in progress update: {e}")
            # Continue execution even if progress reporting fails
            
    def close(self):
        """Override close to handle cleanup safely"""
        try:
            # Only call super().close() if it exists
            if hasattr(super(), 'close'):
                super().close()
        except Exception as e:
            logger.warning(f"Error in tqdm close: {e}")
            
    def __del__(self):
        """Override __del__ to handle cleanup safely"""
        try:
            # Only call super().__del__() if it exists
            if hasattr(super(), '__del__'):
                super().__del__()
        except Exception as e:
            # Silently ignore errors during garbage collection
            pass

class EnhancedDiffusionModel:
    def __init__(
        self,
        temperature: float = 0.65,
        guidance_scale: float = 7.5,
        batch_size: int = 4,
        output_dir: str = "app/generated",
        use_cuda: bool = True,
        controlnet_model: str = "lllyasviel/control_v11p_sd15_scribble",
        base_model: str = "runwayml/stable-diffusion-v1-5",
        scheduler_type: str = "ddim",
        use_latent_diffusion: bool = True,
        cache_models: bool = True,
        dynamic_thresholding: bool = True,
        safety_level: str = "NONE",  # Default to NONE as per requirements
        allow_nsfw: bool = True  # Default to True as per requirements
    ):
        self.temperature = temperature
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.controlnet_model = controlnet_model
        self.base_model = base_model
        self.scheduler_type = scheduler_type
        self.use_latent_diffusion = use_latent_diffusion
        self.cache_models = cache_models
        self.dynamic_thresholding = dynamic_thresholding
        self.safety_level = safety_level
        self.allow_nsfw = allow_nsfw
        
        self.pipe = None
        self.controlnet = None
        self.sketch_classifier = None
        self.clip_model = None
        self.fid_model = None
        self._stored_safety_checker = None
        
        os.makedirs(output_dir, exist_ok=True)
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            logger.info("Initializing model...")
            torch_dtype = torch.float32
            
            # Load ControlNet model
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            
            # Load base Stable Diffusion pipeline with ControlNet
            # Always bypass safety checker as per requirements
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model,
                controlnet=self.controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )

            # Choose appropriate scheduler
            if self.scheduler_type == "ddim":
                # DDIM is faster than default DDPM
                scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
                scheduler.set_timesteps(50)  # Faster inference with fewer steps
            else:
                # Default to Euler Ancestral
                scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            
            self.pipe.scheduler = scheduler
            
            # Enable memory optimizations
            if self.use_cuda:
                if torch.cuda.is_available():
                    device = "cuda"
                    # Use more efficient memory options when possible
                    if importlib.util.find_spec("xformers") is not None:
                        self.pipe.enable_xformers_memory_efficient_attention()
                    else:
                        self.pipe.enable_model_cpu_offload()
                else:
                    device = "cpu"
                    logger.warning("CUDA requested but not available, falling back to CPU")
            else:
                device = "cpu"
                
            self.pipe.to(device)
            
            # Load sketch analysis model
            self._initialize_sketch_recognition()
            
            # Ensure safety is disabled
            self.enable_nsfw_generation()
            
            logger.info(f"Model initialized successfully on {device} using {self.scheduler_type} scheduler")
            logger.warning("⚠️ Safety checker has been disabled for image generation.")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def enable_nsfw_generation(self):
        """Bullet-proof method to enable NSFW image generation"""
        # Method 1: Set safety checker to None
        self.pipe.safety_checker = None
        
        # Method 2: Set requires_safety_checker attribute
        if hasattr(self.pipe, "config"):
            self.pipe.config.requires_safety_checker = False
        
        # Method 3: Use a custom function that always returns False for NSFW detection
        def return_images_without_nsfw_check(clip_input, images):
            return images, [False] * len(images) if isinstance(images, list) else False
            
        self.pipe.safety_checker = return_images_without_nsfw_check
        
        # Method 4: Remove the safety attribute if it exists
        if hasattr(self.pipe, "_safety_check"):
            setattr(self.pipe, "_safety_check", False)
            
        logger.warning("NSFW generation fully enabled. Use responsibly and ethically.")
        self.allow_nsfw = True

    def _initialize_sketch_recognition(self):
        try:
            self.sketch_classifier = pipeline(
                "image-classification", 
                model="microsoft/resnet-50", 
                device=0 if self.use_cuda and torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Sketch recognition model failed to load: {e}")
            raise
    
    def _create_safe_callback_wrapper(self, custom_callback, job_id):
        """
        Create a safe wrapper around custom callbacks to handle tensor arguments
        and prevent errors from propagating to the main pipeline.
        """
        def safe_callback_wrapper(step_idx, t, latents, *args, **kwargs):
            try:
                # Convert tensor arguments to Python scalars if needed
                if hasattr(step_idx, 'item') and callable(getattr(step_idx, 'item')):
                    try:
                        step_idx = step_idx.item()
                    except:
                        step_idx = 0
                
                # Call the original callback with sanitized arguments
                return custom_callback(step_idx, t, latents, *args, **kwargs)
            except Exception as e:
                # Log the error but don't let it crash the pipeline
                logger.warning(f"Error in callback for job {job_id}: {e}")
                return None
        
        return safe_callback_wrapper

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
        styles: Optional[List[str]] = None,
        allow_nsfw: Optional[bool] = None
    ) -> List[str]:
        if job_id is None:
            job_id = str(uuid.uuid4())

        logger.info(f"Starting generation for job {job_id}")
        
        # Register callback with our global registry
        if callback:
            DiffusersProgressCallback.register(job_id, callback)
            # Report initial progress
            callback(0.0, [])

        if temperature is not None:
            self.temperature = temperature
        if guidance_scale is not None:
            self.guidance_scale = guidance_scale

        if seed is not None:
            generator = torch.Generator("cuda" if self.use_cuda else "cpu").manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            generator = None

        try:
            sketch = Image.open(sketch_path).convert("RGB")
            sketch = sketch.resize((512, 512))

            if prompt is None:
                prompt = ""
                
            # Apply style modifiers if provided
            if styles and isinstance(styles, list) and len(styles) > 0:
                style_prompt = ", ".join(styles)
                if prompt:
                    prompt = f"{prompt}, {style_prompt}"
                else:
                    prompt = style_prompt

            # Prepare for batch generation
            batch_count = (num_images + self.batch_size - 1) // self.batch_size
            all_image_paths = []
            
            for batch_idx in range(batch_count):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_images)
                batch_size = end_idx - start_idx
                
                logger.info(f"Generating batch {batch_idx+1}/{batch_count} ({batch_size} images)")
                
                # Create a custom tqdm instance for this batch
                # Wrap in try-except to prevent callback errors from crashing the pipeline
                try:
                    custom_tqdm = lambda *args, **kwargs: GlobalProgressTqdm(
                        *args, 
                        **kwargs,
                        job_id=job_id,
                        image_idx=batch_idx,
                        total_images=batch_count
                    )
                    # Wrap the callback in a safety wrapper to handle tensor arguments
                    custom_tqdm = self._create_safe_callback_wrapper(custom_tqdm, job_id)
                except Exception as e:
                    logger.warning(f"Error creating progress callback: {e}")
                    custom_tqdm = None
                
                # Generate images
                with torch.no_grad():
                    try:
                        # Set safe defaults for callback parameters
                        callback_kwargs = {
                            "prompt": [prompt] * batch_size,
                            "image": [sketch] * batch_size,
                            "negative_prompt": [negative_prompt] * batch_size,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "generator": generator,
                        }
                        
                        # Only add callback if it was created successfully
                        if custom_tqdm is not None:
                            callback_kwargs["callback"] = custom_tqdm
                            callback_kwargs["callback_steps"] = 1  # Ensure callback_steps is always a valid integer
                        
                        output = self.pipe(**callback_kwargs)
                    except Exception as e:
                        logger.error(f"Error in diffusion pipeline: {e}")
                        # Try again without callback if there was an error
                        if custom_tqdm is not None:
                            logger.warning("Retrying without progress callback")
                            output = self.pipe(
                                prompt=[prompt] * batch_size,
                                image=[sketch] * batch_size,
                                negative_prompt=[negative_prompt] * batch_size,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                generator=generator
                            )
                        else:
                            # Re-raise if the error wasn't related to the callback
                            raise
                
                # Save generated images
                batch_image_paths = []
                if save_results:
                    for i, image in enumerate(output.images):
                        global_idx = start_idx + i
                        filename = f"{job_id}_{global_idx}.png"
                        save_path = os.path.join(self.output_dir, filename)
                        
                        # Apply any post-processing here if needed
                        
                        # Save with optimized compression
                        image.save(save_path, format="PNG", optimize=True)
                        batch_image_paths.append(save_path)
                        
                        # Update progress after each image is saved
                        if callback:
                            progress = ((batch_idx * self.batch_size + i + 1) / num_images) * 100
                            all_paths_so_far = all_image_paths + batch_image_paths
                            DiffusersProgressCallback.update(job_id, progress, all_paths_so_far)
                
                all_image_paths.extend(batch_image_paths)
            
            # Clean up
            if callback:
                DiffusersProgressCallback.unregister(job_id)
                
            return all_image_paths

        except Exception as e:
            logger.error(f"Error generating images: {e}")
            if callback:
                DiffusersProgressCallback.unregister(job_id)
            raise
