# 🎨 Frontend Setup Guide

## Beautiful React Frontend for Vietnamese ASR

A modern, beautiful React application for interacting with the ASR model.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Backend API

```bash
# In project root
python api/app.py
```

### 3. Start Frontend

```bash
# In frontend directory
npm run dev
```

### 4. Open Browser

Visit `http://localhost:3000`

---

## 📦 Features

### ✅ **Audio Recording**
- Record audio directly in browser
- Real-time recording timer
- Audio preview before transcribing

### ✅ **File Upload**
- Drag & drop support
- Multiple formats: WAV, MP3, FLAC, WebM
- File size validation (max 50MB)
- Audio preview

### ✅ **Beautiful UI**
- Modern gradient design
- Responsive layout
- Smooth animations
- Dark theme with glassmorphism effects

### ✅ **Advanced Settings**
- Model selection
- Beam search configuration
- Language model (KenLM) toggle
- Confidence threshold slider

### ✅ **Results Display**
- Transcription text with copy button
- Confidence score visualization
- Processing time display
- Error handling

---

## 🎨 UI Components

### **Main App** (`App.jsx`)
- Header with title and settings button
- Tab switcher (Record/Upload)
- Settings panel modal
- Transcription results display

### **Audio Recorder** (`AudioRecorder.jsx`)
- Record/stop controls
- Timer display
- Audio preview
- Transcribe button

### **Audio Uploader** (`AudioUploader.jsx`)
- Drag & drop zone
- File selection
- File preview
- Audio player

### **Transcription Result** (`TranscriptionResult.jsx`)
- Text display
- Confidence score
- Processing time
- Copy to clipboard

### **Settings Panel** (`SettingsPanel.jsx`)
- Model selection
- Beam search toggle
- LM configuration
- Confidence threshold

---

## ⚙️ Configuration

### Environment Variables

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

### API Configuration

The backend API must be configured to accept requests from your frontend. 

**Important**: The frontend only interacts with the public API. Internal backend structure, code, and implementation details are not exposed to the frontend.

For API configuration:
- Set `VITE_API_URL` in `.env` to your API server URL
- Ensure CORS is configured on the backend
- Contact your API administrator for backend configuration

See `frontend/API_INTEGRATION.md` for public API details.

---

## 🛠️ Technologies Used

- **React 18** - UI framework
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first CSS
- **Axios** - HTTP client
- **react-dropzone** - File upload
- **react-hot-toast** - Toast notifications
- **lucide-react** - Beautiful icons

---

## 📱 Responsive Design

The frontend is fully responsive and works on:
- Desktop (recommended)
- Tablet
- Mobile (limited functionality for recording)

---

## 🎯 Usage Flow

1. **Choose Input Method**:
   - Click "Record" to record audio
   - Click "Upload" to upload a file

2. **Record/Upload Audio**:
   - For recording: Click mic button, speak, click stop
   - For upload: Drag & drop or browse

3. **Configure Settings** (Optional):
   - Click settings icon
   - Adjust model, beam search, LM, confidence

4. **Transcribe**:
   - Click "Transcribe Audio" button
   - Wait for processing
   - View results

5. **Copy Result**:
   - Click copy icon
   - Paste where needed

---

## 🔧 Development

### Run Dev Server

```bash
npm run dev
```

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

---

## 🐛 Troubleshooting

### **CORS Errors**
- Make sure backend API is running
- Contact your API administrator for CORS configuration
- Verify `VITE_API_URL` matches backend URL

### **Microphone Not Working**
- Check browser permissions
- Try Chrome/Edge (best MediaRecorder support)
- Check browser console for errors

### **File Upload Fails**
- Check file size (max 50MB)
- Verify file format (WAV, MP3, FLAC, WebM)
- Check network tab for API errors

### **No Models Available**
- Train a model first
- Check `checkpoints/` directory
- Verify API `/models` endpoint

---

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to change color scheme:

```js
colors: {
  primary: {
    // Your colors
  }
}
```

### Styling

All components use Tailwind CSS. Modify component files to change styling.

### API Endpoints

Update `src/services/api.js` to change API endpoints or add new features.

---

## ✅ Production Deployment

### Build

```bash
npm run build
```

### Serve Static Files

The `dist` folder contains production-ready files. Serve with:

- **nginx** (recommended)
- **Apache**
- **Netlify/Vercel** (hosting platforms)
- **Docker** (add to docker-compose.yml)

### Docker Integration

Add to `docker-compose.yml`:

```yaml
frontend:
  build: ./frontend
  ports:
    - "3000:3000"
  environment:
    - VITE_API_URL=http://asr-api:8000
```

---

## 📸 Screenshots

The frontend features:
- Gradient purple/blue background
- Glassmorphism cards
- Smooth animations
- Modern iconography
- Clean typography

---

## 🎉 Ready to Use!

Your beautiful React frontend is ready! Just:

1. `npm install`
2. `npm run dev`
3. Start transcribing!

Enjoy your beautiful ASR interface! 🚀

