# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose a port if your application uses one (e.g., 8000 for Flask or FastAPI)
EXPOSE 5000

# Command to run the application
CMD ["python", "api.py"]
