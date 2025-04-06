FROM python:3.9-slim

WORKDIR /app

# Install dependencies with compatible versions
RUN pip install  \
    pinecone>=3.0.0 \
    python-dotenv>=1.0.0 \
    torch>=2.2.0 \
    sentence-transformers==2.2.2 \
    transformers==4.36.2 \
    huggingface-hub==0.19.4 \
    numpy>=1.24.0

COPY . .

CMD ["python", "vector_db/vectorize.py"] 