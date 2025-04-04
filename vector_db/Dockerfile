FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    pinecone-python==2.2.4 \
    python-dotenv \
    transformers \
    torch \
    sentence-transformers


COPY . .

CMD ["python", "vector_db/vectorize.py"] 