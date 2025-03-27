FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create directories for data
RUN mkdir -p /data/results /data/models /data/scenarios

# Set environment variables
ENV PYTHONPATH=/app
ENV HELIOS_DATA_DIR=/data

# Expose port for web interface (if applicable)
EXPOSE 8050

# Default command
ENTRYPOINT ["helios-sim"]
CMD ["--help"]