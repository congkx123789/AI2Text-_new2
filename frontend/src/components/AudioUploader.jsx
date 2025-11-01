import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileAudio, X } from 'lucide-react'
import { toast } from 'react-hot-toast'

const AudioUploader = ({ onTranscribe, disabled }) => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [audioURL, setAudioURL] = useState(null)

  const validateFile = (file) => {
    // Validate file exists
    if (!file) {
      return { valid: false, error: 'No file selected' }
    }

    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/webm', 'audio/x-wav', 'audio/wave']
    const validExtensions = ['.wav', '.mp3', '.flac', '.webm', '.m4a', '.wma']
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
      return { 
        valid: false, 
        error: 'Invalid file type. Please upload WAV, MP3, FLAC, or WebM format.' 
      }
    }

    // Check file size (max 50MB)
    const maxSize = 50 * 1024 * 1024 // 50MB
    if (file.size > maxSize) {
      const fileSizeMB = (file.size / 1024 / 1024).toFixed(2)
      return { 
        valid: false, 
        error: `File size (${fileSizeMB}MB) exceeds maximum of 50MB` 
      }
    }

    // Check if file is empty
    if (file.size === 0) {
      return { valid: false, error: 'File is empty' }
    }

    return { valid: true }
  }

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Handle rejected files
    if (rejectedFiles && rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0]
      if (rejection.errors) {
        rejection.errors.forEach((error) => {
          if (error.code === 'file-too-large') {
            toast.error('File size exceeds 50MB limit')
          } else if (error.code === 'file-invalid-type') {
            toast.error('Invalid file type. Please use audio files.')
          } else {
            toast.error(`File rejected: ${error.message}`)
          }
        })
      }
      return
    }

    const file = acceptedFiles[0]
    if (file) {
      // Validate file
      const validation = validateFile(file)
      if (!validation.valid) {
        toast.error(validation.error)
        return
      }

      try {
        setSelectedFile(file)
        const url = URL.createObjectURL(file)
        
        // Clean up previous URL if exists
        if (audioURL) {
          URL.revokeObjectURL(audioURL)
        }
        
        setAudioURL(url)
        toast.success('File uploaded successfully')
      } catch (error) {
        console.error('Error processing file:', error)
        toast.error('Error processing file. Please try again.')
      }
    }
  }, [audioURL])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.webm', '.m4a']
    },
    maxFiles: 1,
    disabled
  })

  const removeFile = () => {
    if (audioURL) {
      URL.revokeObjectURL(audioURL)
    }
    setSelectedFile(null)
    setAudioURL(null)
  }

  const handleTranscribe = async () => {
    if (!selectedFile) {
      toast.error('Please select an audio file')
      return
    }
    await onTranscribe(selectedFile)
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Upload Audio</h2>
        <p className="text-white/70">Upload an audio file to transcribe</p>
      </div>

      {/* Dropzone */}
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
            isDragActive
              ? 'border-white bg-white/10'
              : 'border-white/30 hover:border-white/50 hover:bg-white/5'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 text-white/50 mx-auto mb-4" />
          <p className="text-white text-lg font-medium mb-2">
            {isDragActive ? 'Drop audio file here' : 'Drag & drop audio file here'}
          </p>
          <p className="text-white/60 text-sm">or click to browse</p>
          <p className="text-white/40 text-xs mt-2">
            Supports: WAV, MP3, FLAC, WebM (Max 50MB)
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* File Info */}
          <div className="bg-white/10 rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <FileAudio className="w-8 h-8 text-white/70" />
              <div>
                <p className="text-white font-medium">{selectedFile.name}</p>
                <p className="text-white/60 text-sm">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onClick={removeFile}
              disabled={disabled}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Audio Preview */}
          {audioURL && (
            <audio src={audioURL} controls className="w-full rounded-lg" />
          )}

          {/* Transcribe Button */}
          <button
            onClick={handleTranscribe}
            disabled={disabled}
            className="w-full py-4 px-6 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-xl transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Transcribe Audio
          </button>

          {/* Change File Button */}
          <button
            onClick={removeFile}
            disabled={disabled}
            className="w-full py-3 px-6 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl transition-all disabled:opacity-50"
          >
            Choose Different File
          </button>
        </div>
      )}
    </div>
  )
}

export default AudioUploader

