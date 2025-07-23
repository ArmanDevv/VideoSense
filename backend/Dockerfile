FROM node:18-slim

# Install dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev \
    ffmpeg libsndfile1 \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

WORKDIR /app

# Install Node.js dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Create Python environment
RUN python3 -m venv venv
RUN ./venv/bin/pip install --upgrade pip --no-cache-dir

# Install Python packages and remove cache
COPY requirements.txt ./
RUN ./venv/bin/pip install --prefer-binary --no-cache-dir -r requirements.txt \
    && find /app/venv -name "*.pyc" -delete \
    && find /app/venv -name "__pycache__" -delete

# Remove build dependencies after installation
RUN apt-get remove -y build-essential gcc python3-dev \
    && apt-get autoremove -y \
    && apt-get autoclean

# Copy application code
COPY . .

EXPOSE 5000
CMD ["node", "server.js"]
