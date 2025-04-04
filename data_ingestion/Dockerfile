FROM python:3.9-slim

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Install dependencies
RUN pip install --no-cache-dir python-dotenv==0.19.0 gitpython==3.1.24

COPY . .

CMD ["python", "data_ingestion/ingest.py"]