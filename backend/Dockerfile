FROM node:18-slim

# Install dependencies including cookie handling tools
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev \
    ffmpeg libsndfile1 \
    curl wget git build-essential \
    libssl-dev libffi-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Node.js dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Create Python environment
RUN python3 -m venv venv
RUN ./venv/bin/pip install --upgrade pip --no-cache-dir

# Install Python packages
COPY requirements.txt ./
RUN ./venv/bin/pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000
CMD ["node", "server.js"]
