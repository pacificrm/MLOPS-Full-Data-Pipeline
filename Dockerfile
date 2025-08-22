# Use a slim Python base image consistent with the project's dependencies
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code, including app.py and model.joblib
COPY . .

# Command to run the FastAPI application using uvicorn
# This tells uvicorn to run the 'app' instance from the 'app.py' file.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

