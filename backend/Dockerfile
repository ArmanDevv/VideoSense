FROM node:18-slim

# Install Python, system dependencies, and build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Node.js dependencies
COPY package*.json ./
RUN npm ci

# Create Python virtual environment
RUN python3 -m venv venv

# Upgrade pip and install wheel support
RUN ./venv/bin/pip install --upgrade pip wheel setuptools

# Copy and install Python dependencies with prefer-binary flag
COPY requirements.txt ./
RUN ./venv/bin/pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000
CMD ["node", "server.js"]
