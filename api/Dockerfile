FROM python:3.9-slim

WORKDIR /app

# Install required dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    python-dotenv>=1.0.0 \
    pinecone>=3.0.0 \
    google-generativeai>=0.3.0

COPY . .

EXPOSE 5000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]    