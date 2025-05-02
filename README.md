# SketchXpress Backend

This repository contains the backend service for SketchXpress, an application designed to transform input sketches into detailed images using AI diffusion models.

## Features

- **Sketch-to-Image Generation:** Utilizes Stable Diffusion and ControlNet (specifically `lllyasviel/sd-controlnet-scribble`) to generate images based on input sketches and optional text prompts.
- **Asynchronous Processing:** Employs a Celery/Redis job queue to handle computationally intensive image generation tasks asynchronously, keeping the API responsive.
- **Job Status Tracking:** Provides an endpoint to check the status and progress of ongoing generation tasks.
- **Automatic File Cleanup:** Includes a scheduled task (via Celery Beat) to automatically delete uploaded sketches and generated images older than 1 hour, managing disk space.
- **Optimized Performance:** Incorporates various optimizations for faster inference and efficient resource usage, including:
  - Faster inference scheduler (`EulerAncestralDiscreteScheduler`).
  - Optimized PNG compression.
  - CPU+GPU offloading for efficient VRAM usage.
  - Dockerized deployment with model caching and GPU support.

## Architecture

The backend consists of several containerized services managed by Docker Compose:

1.  **`redis`:** A Redis instance acting as the message broker and result backend for Celery.
2.  **`backend`:** A FastAPI application providing the REST API endpoints for submitting generation jobs and checking status.
3.  **`worker`:** A Celery worker process that consumes tasks from the queue, loads the AI models (Stable Diffusion + ControlNet), performs the image generation using GPU acceleration, and updates job status.
4.  **`beat`:** A Celery Beat scheduler process responsible for triggering periodic tasks, such as the hourly file cleanup.

## Requirements

- **Operating System:** Linux distribution compatible with Docker and NVIDIA drivers (e.g., Ubuntu 20.04+).
- **Hardware:**
  - **CPU:** Multi-core CPU.
  - **RAM:** 16GB+ recommended.
  - **GPU:** NVIDIA GPU with CUDA support and sufficient VRAM (8GB+ recommended, 10GB+ ideal) is **highly recommended** for performance.
  - **Storage:** Disk space for Docker images, model cache (~10-20GB), and temporary files.
- **Software:**
  - Docker Engine
  - Docker Compose
  - NVIDIA Container Toolkit (for GPU support)

## Setup and Deployment

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> sketchxpress-backend
    cd sketchxpress-backend
    ```
2.  **Prerequisites:** Ensure Docker, Docker Compose, and NVIDIA Container Toolkit are installed on your host machine.
3.  **Build the Docker Image:**
    ```bash
    docker compose build
    ```
4.  **Create Local Directories (Optional but Recommended):**
    ```bash
    mkdir -p ./app/uploads
    mkdir -p ./app/generated
    ```
5.  **Start the Services:**
    ```bash
    docker compose up -d
    ```
    - This will start the `redis`, `backend`, `worker`, and `beat` containers.
    - On the first run, the `worker` container will download the necessary AI models, which may take some time. These models will be cached in the `hf_cache` Docker volume for subsequent runs.

## Usage

### API Endpoints

- **`POST /api/generate`**: Submit a new sketch-to-image generation job.
  - **Form Data:**
    - `sketch`: The input sketch image file (e.g., PNG, JPG).
    - `prompt` (optional): A text description of the desired output.
    - `temperature` (optional, default: 0.65): Controls randomness.
    - `guidance_scale` (optional, default: 7.5): How strongly the prompt guides generation.
    - `num_images` (optional, default: 1): Number of images to generate.
    - `steps` (optional, default: 30): Number of diffusion steps (higher = more detail, slower).
    - `seed` (optional): Seed for reproducibility.
  - **Response:**
    ```json
    { "status": "queued", "job_id": "<unique_job_identifier>" }
    ```
- **`GET /api/status/{job_id}`**: Check the status of a generation job.
  - **Path Parameter:**
    - `job_id`: The unique identifier returned by the `/api/generate` endpoint.
  - **Response (Example - In Progress):**
    ```json
    {
      "job_id": "<unique_job_identifier>",
      "status": "PROGRESS",
      "progress": 50.0,
      "images": [],
      "message": "Task is in progress."
    }
    ```
  - **Response (Example - Success):**
    ```json
    {
      "job_id": "<unique_job_identifier>",
      "status": "completed",
      "progress": 100.0,
      "images": ["/generated/<job_id>_0.png"],
      "message": "Task completed successfully."
    }
    ```
    _(Note: Image paths are relative to the API base URL, e.g., `http://localhost:8000/generated/...`)_
- **`GET /docs`**: Access the interactive FastAPI documentation (Swagger UI).
- **`GET /redoc`**: Access alternative API documentation (ReDoc).

### Example (using curl)

1.  **Submit Job:**
    ```bash
    curl -X POST -F "sketch=@./app/uploads/demoCar.png" -F "prompt=a red sports car" http://localhost:8000/api/generate
    # Note the returned job_id
    ```
2.  **Check Status (replace `<job_id>`):**
    ```bash
    curl http://localhost:8000/api/status/<job_id>
    ```

## Configuration

- **Docker Compose (`docker-compose.yml`):** Defines the services, volumes, network, and GPU configuration.
- **Celery (`app/worker.py`):** Configures the task queue, worker concurrency, result backend, and the scheduled cleanup task (`beat_schedule`).
- **Model (`app/model/enhanced_model.py`):** Contains model loading logic and generation parameters.
- **API (`app/api.py`):** Configures FastAPI, CORS settings, and API endpoints.

## File Cleanup

The `beat` service runs an hourly task defined in `app/worker.py` (`cleanup_old_files_task`). This task checks the `./app/uploads` and `./app/generated` directories (relative to the project root on the host, mounted into containers) and deletes any files whose modification time is older than 1 hour (3600 seconds).
