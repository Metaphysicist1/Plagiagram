FROM python:3.9-slim

WORKDIR /app

# Install required dependencies
RUN pip install --no-cache-dir \
    flask>=3.0.0 \
    werkzeug>=3.0.0 \
    python-dotenv>=1.0.0 \
    pinecone>=3.0.0 \
    sentence-transformers>=2.5.0 \
    torch>=2.2.0 \
    transformers>=4.37.0 \
    google-generativeai>=0.3.0

COPY . .

EXPOSE 5000

CMD ["python", "api/app.py"]    