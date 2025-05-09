FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (only if needed; skip if pure Python)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary app files and folders
COPY src/__init__.py /app/src/__init__.py
COPY src/logger/__init__.py /app/src/logger/__init__.py
COPY src/data/__init__.py /app/src/data/__init__.py
COPY src/data/data_transformation.py /app/src/data/data_transformation.py
COPY flask_app/app.py /app/app.py
COPY models/model/model.cbm /app/models/model/model.cbm
COPY models/preprocessor/preprocessing_pipeline.pkl /app/models/preprocessor/preprocessing_pipeline.pkl
COPY flask_app/templates /app/templates
COPY flask_app/static /app/static

# Expose port
EXPOSE 5000

# Run app
#CMD ["python", "app.py"]

# Production option (optional; comment out above CMD if using this)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
