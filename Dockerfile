# Stage 1: Build dependencies
FROM python:3.12.2-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# Create a virtual environment
RUN python -m venv /opt/venv
# Activate virtual environment and install requirements
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel
# Ensure cache directory exists and is writable (though running as root should be fine)
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME
RUN pip install -r requirements.txt

# Stage 2: Create final image
FROM python:3.12.2-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy app files
COPY ./app ./app

# Ensure cache directory exists in final image (for structure, volume mount will handle data)
RUN mkdir -p $HF_HOME

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

