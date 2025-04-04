FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The file is at /app/vector_db/vectorize.py, but we're already in /app
CMD ["python", "vector_dbvectorize.py"] 