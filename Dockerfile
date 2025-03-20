FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Mount dataset, checkpoints, and predictions folders
VOLUME ["/app/src", "/app/checkpoints", "/app/predictions"]

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variables to speed up Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Define the command to run the Streamlit UI
CMD ["streamlit", "run", "/app/app.py"]
