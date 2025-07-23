FROM node:18-slim

# Install Python, system dependencies, and build tools INCLUDING libsndfile1
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    ffmpeg \
    curl \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    pkg-config \
    git \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using official installer
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Update PATH to include Rust binaries
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy and install Node.js dependencies
COPY package*.json ./
RUN npm ci

# Create Python virtual environment
RUN python3 -m venv venv

# Upgrade pip in virtual environment
RUN ./venv/bin/pip install --upgrade pip

# Copy and install Python dependencies in virtual environment
COPY requirements.txt ./
RUN ./venv/bin/pip install -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000
CMD ["node", "server.js"]
