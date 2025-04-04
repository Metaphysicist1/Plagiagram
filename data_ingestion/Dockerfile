FROM python:3.9-slim

# Set initial working directory
WORKDIR /app

# Copy everything from build context to /app/
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Run the ingest.py script
ENTRYPOINT ["/bin/bash"]
# CMD ["python", "ingest.py"] 