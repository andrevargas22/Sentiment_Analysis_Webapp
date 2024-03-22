# Use the official Python image as a base image
FROM python:3.9-slim

# Set environment variables
ENV PORT 8000
ENV HOST 0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
COPY cache/sentiment_analysis/vocabulary.pkl /app/cache/sentiment_analysis/vocabulary.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port on which the FastAPI app runs
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

