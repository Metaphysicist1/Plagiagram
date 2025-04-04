FROM python:3.9-slim

# Create temp directory and set permissions
RUN mkdir -p /tmp && chmod 1777 /tmp

# Install git
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated git

# Set initial working directory
WORKDIR /app

# Create requirements.txt file directly in the container
RUN echo "python-dotenv==0.19.0\ngitpython==3.1.24" > requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from build context to /app/
COPY . /app/

# Change to data_ingestion directory
WORKDIR /app/data_ingestion

# Set bash as entrypoint
ENTRYPOINT ["/bin/bash"]

# Default command
CMD ["-c", "python ingest.py"] 