FROM python:3.9-slim

WORKDIR /app

# Install dependencies directly
RUN pip install --no-cache-dir pinecone numpy pandas scikit-learn dotenv google-generativeai

COPY . .

# The file is at /app/vector_db/vectorize.py, but we're already in /app
CMD ["python", "vector_db/vectorize.py"] 