FROM python:3.9-slim

WORKDIR /app

# Install required dependencies
RUN pip install --no-cache-dir \
    flask==2.0.1 \
    werkzeug==2.0.1 \
    python-dotenv \
    pinecone \
    google-generativeai

COPY . .

EXPOSE 5000

CMD ["python", "api/app.py"] 