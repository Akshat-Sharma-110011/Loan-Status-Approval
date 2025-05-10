FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directory structure
RUN mkdir -p /app/src/data
RUN mkdir -p /app/models/model
RUN mkdir -p /app/models/preprocessor
RUN mkdir -p /app/logs
RUN mkdir -p /app/templates
RUN mkdir -p /app/static

# Create required __init__.py files
RUN touch /app/src/__init__.py
RUN touch /app/src/data/__init__.py

# Install Python dependencies
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files with explicit paths
COPY src/data/data_transformation.py /app/src/data/data_transformation.py
COPY models/model/ /app/models/model/
COPY models/preprocessor/ /app/models/preprocessor/
COPY flask_app/app.py /app/app.py
COPY flask_app/templates/ /app/templates/
COPY flask_app/static/ /app/static/

# Make sure application directory is in PYTHONPATH
ENV PYTHONPATH="/app"

# Expose port
EXPOSE 5000

# Create a debugging script to print environment info
RUN echo '#!/bin/bash\n\
echo "===== Directory Structure =====" \n\
find /app -type f | sort\n\
echo "===== Python Path =====" \n\
echo $PYTHONPATH \n\
echo "===== Python Version =====" \n\
python --version \n\
echo "===== Installed Packages =====" \n\
pip list \n\
' > /app/debug_info.sh

RUN chmod +x /app/debug_info.sh

# Run the application with gunicorn directly
CMD ["/bin/bash", "-c", "/app/debug_info.sh && python -m gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app"]