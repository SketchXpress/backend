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

logger = logging.getLogger("enhanced_model")
logging.basicConfig(level=logging.INFO)

# Add this at the module level, before the EnhancedDiffusionModel class
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
        super().__init__(*args, **kwargs)
        
    def update(self, n=1):
        """Override update to report progress to our global registry"""
        super().update(n)
        
        # Only report if we have a job_id
        if self.job_id and self.total:
            # Calculate overall progress as percentage
            progress = min(100.0, (
                (self.image_idx * 100.0 / self.total_images) + 
                (self.n * 100.0 / (self.total * self.total_images))
            ))
            
            # Report to the global registry
            DiffusersProgressCallback.update(self.job_id, progress)

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
        safety_level: str = "MEDIUM",  # Options: "NONE", "WEAK", "MEDIUM", "STRONG", "MAX"
        allow_nsfw: bool = False  # New parameter for NSFW generation
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
            if self.allow_nsfw:
                # Completely bypass safety checker for NSFW generation
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    self.base_model,
                    controlnet=self.controlnet,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            else:
                # Load safety checker and feature extractor
                safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker",
                    torch_dtype=torch_dtype
                )
                feature_extractor = CLIPFeatureExtractor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    self.base_model,
                    controlnet=self.controlnet,
                    torch_dtype=torch_dtype,
                    safety_checker=safety_checker,
                    feature_extractor=feature_extractor,
                    use_safetensors=True
                )
                
                # Apply safety level configurations
                if self.safety_level == "NONE":
                    self.set_safety_checker_enabled(False)
                elif self.safety_level != "MAX":
                    self.create_custom_safety_checker()

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
            
            # Load evaluation models if needed
            self._initialize_evaluation_models()
            
            # Load sketch analysis model
            self._initialize_sketch_recognition()
            
            logger.info(f"Model initialized successfully on {device} using {self.scheduler_type} scheduler")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def create_custom_safety_checker(self):
        """Create a custom safety checker that detects but doesn't black out images"""
        original_safety_checker = self.pipe.safety_checker
        feature_extractor = self.pipe.feature_extractor
        
        def custom_safety_checker(clip_input, images):
            # Return original images but still detect NSFW content
            if original_safety_checker is not None and feature_extractor is not None:
                safety_checker_input = feature_extractor(images, return_tensors="pt").to(original_safety_checker.device)
                _, has_nsfw_concepts = original_safety_checker(
                    images=safety_checker_input.pixel_values,
                    clip_input=clip_input
                )
                # Key difference: return original images, not black ones
                return images, has_nsfw_concepts
            else:
                return images, [False] * len(images)
        
        # Replace the safety checker with our custom version
        self.pipe.safety_checker = custom_safety_checker

    def set_safety_checker_enabled(self, enabled: bool = True):
        """Enable or disable the safety checker"""
        if not enabled:
            logger.warning("Safety checker being disabled. Ensure you're using this responsibly.")
            # Store but don't use the safety checker
            self._stored_safety_checker = self.pipe.safety_checker
            self.pipe.safety_checker = None
            # These additional settings ensure the safety checker stays disabled
            setattr(self.pipe, "requires_safety_checker", False)
            if hasattr(self.pipe.config, "requires_safety_checker"):
                self.pipe.config.requires_safety_checker = False
        else:
            # Restore the safety checker if it was previously stored
            if hasattr(self, "_stored_safety_checker") and self._stored_safety_checker is not None:
                self.pipe.safety_checker = self._stored_safety_checker
                if hasattr(self.pipe.config, "requires_safety_checker"):
                    self.pipe.config.requires_safety_checker = True
            elif self.pipe.safety_checker is None and not self.allow_nsfw:
                # Re-initialize if needed
                self.pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker",
                    torch_dtype=self.pipe.dtype
                )
                self.pipe.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )

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
            
    def _initialize_evaluation_models(self):
        """Load models for evaluating generation quality (CLIP, FID)"""
        # Only load these if specifically requested to save memory
        pass
    
    def _apply_dynamic_thresholding(self, latents, threshold_pct=0.95):
        """Apply dynamic thresholding to prevent oversaturation"""
        if not self.dynamic_thresholding:
            return latents
            
        # Calculate dynamic thresholding values
        s = torch.quantile(
            torch.abs(latents).reshape(latents.shape[0], -1),
            threshold_pct,
            dim=1
        )
        s = torch.maximum(s, torch.ones_like(s))
        s = s.reshape(-1, 1, 1, 1)
        
        # Apply dynamic thresholding
        latents = torch.clamp(latents, -s, s) / s
        return latents

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
            
    def evaluate_generation(self, images: List[Image.Image], prompts: List[str]) -> Dict:
        """Evaluate the quality of generated images using metrics like CLIP score"""
        # This would implement evaluation metrics - left as a placeholder
        return {"quality_score": 0.8}

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
        
        # Check if NSFW generation should be enabled for this specific job
        if allow_nsfw is not None and allow_nsfw and not self.allow_nsfw:
            self.enable_nsfw_generation()
        
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
                analysis = self.analyze_sketch(sketch_path)
                prompt = f"high quality, detailed {analysis['top_class']}, professional"

            logger.info(f"Using prompt: {prompt}")
            
            # Report progress after sketch loading and analysis
            if callback:
                callback(5.0, [])
                
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
        
        # Create a TqdmCallback factory function
        def tqdm_with_job_id(*args, **kwargs):
            kwargs["job_id"] = job_id
            kwargs["image_idx"] = getattr(tqdm_with_job_id, "image_idx", 0)
            kwargs["total_images"] = num_images
            return GlobalProgressTqdm(*args, **kwargs)
        
        # Store original tqdm
        import diffusers.utils
        original_tqdm = diffusers.utils.tqdm
        
        try:
            # Replace diffusers tqdm with our version
            diffusers.utils.tqdm = tqdm_with_job_id
            
            for i in range(num_images):
                try:
                    # Update the image index for progress tracking
                    tqdm_with_job_id.image_idx = i
                    
                    styled_prompt = f"{prompt}, {styles[i % len(styles)]}"
                    
                    # Generate the image
                    output = self.pipe(
                        prompt=styled_prompt,
                        image=sketch,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        generator=generator,
                        return_dict=True
                    )

                    # Handle potentially NSFW content
                    images = output.images
                    nsfw_detected = getattr(output, "nsfw_content_detected", [False] * len(images))

                    for j, (img, is_nsfw) in enumerate(zip(images, nsfw_detected)):
                        img_path = os.path.join(self.output_dir, f"{job_id}_{i}_{j}.png")
                        
                        if is_nsfw:
                            logger.warning(f"NSFW content detected in image {i}_{j}")
                            
                            # Apply different handling based on safety level
                            if self.safety_level == "STRONG" and not self.allow_nsfw:
                                # Apply heavy blur to NSFW content
                                img = img.filter(ImageFilter.GaussianBlur(radius=30))
                            elif self.safety_level == "MEDIUM" and not self.allow_nsfw:
                                # Apply light blur
                                img = img.filter(ImageFilter.GaussianBlur(radius=10))
                            # WEAK and NONE levels keep image as is
                        
                        img.save(img_path, compress_level=1)
                        image_paths.append(img_path)

                    # Update images in our registry after each image
                    if callback:
                        DiffusersProgressCallback.update(job_id, 
                                                      ((i+1) * 100.0) / num_images, 
                                                      image_paths)
                        
                except Exception as e:
                    logger.error(f"Error generating image {i+1}: {e}")
                    continue
                
        finally:
            # Restore original tqdm
            diffusers.utils.tqdm = original_tqdm
            
            # Ensure final progress is reported
            if callback:
                callback(100.0, image_paths)
                
            # Clean up the registry
            DiffusersProgressCallback.unregister(job_id)

        logger.info(f"Completed {len(image_paths)} images for job {job_id}")
        return image_paths

    def rank_images(self, image_paths: List[str]) -> List[str]:
        """Sort generated images by estimated quality"""
        # Could implement more sophisticated ranking using CLIP or other metrics
        return image_paths
        
    def cleanup(self):
        """Release model resources when done"""
        if not self.cache_models:
            self.pipe = None
            self.controlnet = None
            self.sketch_classifier = None
            torch.cuda.empty_cache()
