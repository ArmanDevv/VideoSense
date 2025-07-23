FROM node:18-slim

# Install Python and virtual environment support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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

# Start the application (Node.js doesn't need venv)
CMD ["node", "server.js"]
