const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const bcrypt = require('bcryptjs');
const cors = require('cors');
require('dotenv').config(); 
const app = express();
app.use(cors({
  origin: [
    'http://localhost:3000',           // Local development
    'http://localhost:5173',           // Local Vite dev server  
    'https://videosense.vercel.app',   // Your Vercel deployment
    'https://*.vercel.app'             // Any Vercel preview deployments
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept']
}));
// Handle preflight requests for all routes
app.options('*', cors());
app.use(express.json());
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI);

// User Schema
const User = mongoose.model('User', {
    name: String,
    email: { type: String, unique: true },
    password: String,
    company: String
});

// REGISTER Route - Creates NEW user in database
app.post('/api/register', async (req, res) => {
    try {
        const { name, email, password, company } = req.body;
        
        console.log('🔵 REGISTER attempt for:', email);
        
        // Check if user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            console.log('❌ Registration failed - user already exists');
            return res.status(400).json({ message: 'User with this email already exists!' });
        }

        // Hash password and CREATE new user in database
        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = new User({ 
            name, 
            email, 
            password: hashedPassword, 
            company: company || '' 
        });
        
        // SAVE to MongoDB
        await newUser.save();
        console.log('✅ New user created in database:', name);

        res.json({ 
            message: 'Account created successfully!',
            user: { name: newUser.name, email: newUser.email, company: newUser.company }
        });
    } catch (error) {
        console.log('❌ Registration error:', error);
        res.status(500).json({ message: 'Registration failed!' });
    }
});

// LOGIN Route - Checks EXISTING user in database
app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        
        console.log('🔴 LOGIN attempt for:', email);
        
        // FIND existing user in database
        const user = await User.findOne({ email });
        if (!user) {
            console.log('❌ Login failed - no user found with this email');
            return res.status(400).json({ message: 'No account found with this email!' });
        }

        // CHECK password against stored hash
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            console.log('❌ Login failed - wrong password');
            return res.status(400).json({ message: 'Wrong password!' });
        }

        console.log('✅ Login successful for:', user.name);
        res.json({ 
            message: 'Login successful!',
            user: { name: user.name, email: user.email, company: user.company }
        });
    } catch (error) {
        console.log('❌ Login error:', error);
        res.status(500).json({ message: 'Login failed!' });
    }
});

app.post('/api/analyze', async (req, res) => {
  const { videoUrl } = req.body;
  
  console.log(`Starting analysis for: ${videoUrl}`);
  
  const py = spawn('python3', [
    path.join(__dirname, 'python', 'inference.py'),
    '--video_url', videoUrl
  ], {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  let stdoutData = '';
  let stderrData = '';
  let responseAlreadySent = false; // ✅ Track if response was sent
  
  // ✅ Set timeout and store the reference
  const timeoutId = setTimeout(() => {
    if (!responseAlreadySent && !py.killed) {
      responseAlreadySent = true;
      py.kill('SIGTERM');
      res.status(500).json({ error: 'Python script timeout' });
    }
  }, 120000); // 2 minutes timeout
  
  py.stdout.on('data', (chunk) => {
    stdoutData += chunk.toString();
  });
  
  py.stderr.on('data', (chunk) => {
    stderrData += chunk.toString();
    console.log('Python stderr:', chunk.toString());
  });

  py.on('close', (code) => {
    // ✅ Clear the timeout immediately when process closes
    clearTimeout(timeoutId);
    
    // ✅ Check if response was already sent
    if (responseAlreadySent) {
      return;
    }
    
    console.log(`Python script exited with code: ${code}`);
    
    if (code !== 0) {
      responseAlreadySent = true;
      return res.status(500).json({ 
        error: 'Python script failed', 
        exitCode: code,
        stderr: stderrData,
        stdout: stdoutData
      });
    }
    
    try {
      const lines = stdoutData.split('\n');
      let jsonData = '';
      
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
          jsonData = trimmed;
          break;
        }
      }
      
      if (!jsonData) {
        throw new Error('No JSON data found in Python output');
      }
      
      const result = JSON.parse(jsonData);
      responseAlreadySent = true;
      res.json(result);
      
    } catch (parseError) {
      console.error('JSON Parse Error:', parseError.message);
      responseAlreadySent = true;
      res.status(500).json({ 
        error: 'Failed to parse Python output',
        parseError: parseError.message,
        rawOutput: stdoutData.substring(0, 1000)
      });
    }
  });

  py.on('error', (error) => {
    // ✅ Clear timeout on error too
    clearTimeout(timeoutId);
    
    if (!responseAlreadySent) {
      console.error('Python process error:', error);
      responseAlreadySent = true;
      res.status(500).json({ 
        error: 'Failed to start Python process',
        details: error.message 
      });
    }
  });
});




app.listen(5000, () => {
    console.log('Server running on port 5000');
});
