# Vietnamese ASR Frontend

Beautiful React frontend for the Vietnamese ASR (Speech-to-Text) system.

## Features

- 🎤 **Audio Recording** - Record audio directly in the browser
- 📁 **File Upload** - Upload audio files (WAV, MP3, FLAC, WebM)
- 🎨 **Modern UI** - Beautiful, responsive design with gradient backgrounds
- ⚙️ **Advanced Settings** - Configure beam search, language model, confidence threshold
- 📊 **Results Display** - Shows transcription with confidence scores and processing time
- 📋 **Copy to Clipboard** - Easy copying of transcription results

## Prerequisites

- Node.js 16+ and npm/yarn
- Backend API running on `http://localhost:8000` (or configure `VITE_API_URL`)

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Environment Variables

Create a `.env` file in the `frontend` directory:

```env
VITE_API_URL=http://localhost:8000
```

## Usage

1. **Ensure backend API is running** on the configured URL (default: `http://localhost:8000`)

2. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open in browser**: `http://localhost:3000`

4. **Record or upload audio** and click "Transcribe Audio"

> **Note**: Backend API must be running separately. See API documentation for backend setup.

## Features

### Audio Recording
- Click the microphone button to start recording
- Click the square button to stop
- View recording time and preview
- Transcribe the recorded audio

### File Upload
- Drag and drop audio files
- Or click to browse
- Supports: WAV, MP3, FLAC, WebM (max 50MB)
- Preview audio before transcribing

### Settings
- **Model Selection**: Choose from available models
- **Beam Search**: Enable/disable with configurable beam width
- **Language Model**: Enable KenLM for better accuracy
- **Confidence Threshold**: Filter low-confidence predictions

### Results
- Display transcription text
- Show confidence score with visual indicator
- Display processing time
- Copy to clipboard functionality

## Technologies

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **react-dropzone** - File upload
- **react-hot-toast** - Notifications
- **lucide-react** - Icons

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari

**Note**: Audio recording requires a browser that supports MediaRecorder API.

## Troubleshooting

### CORS Issues
If you encounter CORS errors, contact your backend administrator. The API server must be configured to allow requests from your frontend origin.

### Microphone Permissions
If recording doesn't work, check browser permissions for microphone access. Some browsers require explicit permission.

### API Connection
Ensure the backend API is running and accessible at the configured URL. Check the API URL in your `.env` file.

## License

Same as main project.

