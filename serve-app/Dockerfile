FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the project
COPY . .

# Expose the port
EXPOSE 5000

# Create a volume for the model
VOLUME /model

# Set environment variable
ENV RUNNING_IN_DOCKER="true"

# Run the application
CMD ["python", "app.py"]
