FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 5000

# Add the CMD instruction to run the Flask app
CMD ["python", "api/app.py"]