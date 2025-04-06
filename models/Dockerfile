FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install \
    google-generativeai>=0.3.0 \
    pinecone>=3.0.0 \
    numpy>=1.24.0 \
    python-dotenv>=1.0.0

# Copy the models directory
COPY models/ .

# Set environment variables
ENV PYTHONPATH=/app 