# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all files from your repo into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run expects the app to listen on port 8080
EXPOSE 8080

# The command to start the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
