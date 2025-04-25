'''
Centralized error logging, fallback messages, and HTTP exceptions
'''
import logging
import traceback
from fastapi import HTTPException

# Setup logging
logger = logging.getLogger("model_error_handler")
logging.basicConfig(level=logging.INFO)

class ModelErrorHandler:
    """
    Error handler for AI model initialization and usage.
    Provides graceful fallback and detailed error reporting.
    """

    @staticmethod
    def handle_import_error(e, module_name):
        msg = f"Could not import {module_name}: {str(e)}"
        logger.error(msg)
        logger.error(traceback.format_exc())

        if "matplotlib" in str(e):
            logger.error("Install with: pip install matplotlib")
        elif "scikit-image" in str(e) or "skimage" in str(e):
            logger.error("Install with: pip install scikit-image")
        elif "cv2" in str(e):
            logger.error("Install with: pip install opencv-python-headless")

        return msg

    @staticmethod
    def handle_model_initialization_error(e, model_name):
        msg = f"Failed to initialize {model_name}: {str(e)}"
        logger.error(msg)
        logger.error(traceback.format_exc())

        if "CUDA" in str(e) or "GPU" in str(e):
            logger.error("Check if CUDA is properly installed and the GPU has enough memory.")

        return msg

    @staticmethod
    def handle_generation_error(e, job_id=None):
        msg = f"Error during image generation"
        if job_id:
            msg += f" for job {job_id}"
        msg += f": {str(e)}"

        logger.error(msg)
        logger.error(traceback.format_exc())

        if "out of memory" in str(e).lower():
            logger.error("Try reducing batch size or image resolution.")
        elif "timeout" in str(e).lower():
            logger.error("Operation timed out. Try simplifying the generation prompt.")

        return msg

    @staticmethod
    def create_fallback_response(error_type, error_msg, job_id=None):
        response = {
            "status": "error",
            "error_type": error_type,
            "message": error_msg
        }

        if job_id:
            response["job_id"] = job_id

        return response

    @staticmethod
    def raise_http_exception(status_code, error_type, error_msg):
        raise HTTPException(
            status_code=status_code,
            detail={
                "status": "error",
                "error_type": error_type,
                "message": error_msg
            }
        )
