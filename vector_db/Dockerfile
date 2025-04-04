FROM python:3.9-slim

WORKDIR /app

# Install network troubleshooting tools
RUN apt-get update && apt-get install -y dnsutils iputils-ping

# Copy the entire project
COPY . /app/

# Install dependencies from the root requirements file
RUN pip install --no-cache-dir -r requirements.txt

# The file is at /app/vector_db/vectorize.py, but we're already in /app
CMD ["python", "vector_db/vectorize.py"] 