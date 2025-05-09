FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs

# Copy application files
COPY src /app/src
COPY models /app/models
COPY flask_app/app.py /app/app.py
COPY flask_app/templates /app/templates
COPY flask_app/static /app/static

# Make sure application directory is in PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose port
EXPOSE 5000

# Create a wrapper script to start the application
RUN echo '#!/bin/bash\n\
export PYTHONPATH="/app:${PYTHONPATH}"\n\
cd /app\n\
python -m gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app\n' > /app/start.sh

RUN chmod +x /app/start.sh

# Run the application using the wrapper script
CMD ["/app/start.sh"]