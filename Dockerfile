FROM python:3.12.2-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy app files
COPY ./app ./app

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
