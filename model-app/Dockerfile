FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the project
COPY . .

# Create a volume for the model and data
VOLUME /model
VOLUME /data

# Set environment variable
ENV RUNNING_IN_DOCKER=true

# Run the application
CMD ["python", "app.py"]
