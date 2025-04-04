FROM python:3.9-slim

WORKDIR /app

# Copy the entire project
COPY . /app/

# Install dependencies from the root requirements file
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# The file is at /app/api/app.py, but we're already in /app
CMD ["python", "app.py"] 