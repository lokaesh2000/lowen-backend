# Use a lightweight Python image
FROM python:3.11-slim

# Install system libraries required for OpenCV and Graphics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a place for our code
WORKDIR /app

# Copy the grocery list and install everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your AI models and your code into the server
COPY . .

# Tell the server to listen for web traffic on port 8000
EXPOSE 8000

# Start the 'Waiter' (FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]