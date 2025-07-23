FROM node:18-slim

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Node.js dependencies
COPY package*.json ./
RUN npm ci

# Copy and install Python dependencies  
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000
CMD ["node", "server.js"]
