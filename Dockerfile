# Use official Python base image
FROM python:3.10.10

# Set environment variables to optimize behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure ports are exposed
EXPOSE 8000

# Start the app using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]

