import { useState, useRef, useEffect } from 'react'
import { Mic, Square, RotateCcw } from 'lucide-react'
import { toast } from 'react-hot-toast'

const AudioRecorder = ({ onTranscribe, disabled }) => {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState(null)
  const [audioURL, setAudioURL] = useState(null)
  const mediaRecorderRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (audioURL) {
        URL.revokeObjectURL(audioURL)
      }
    }
  }, [audioURL])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      const chunks = []
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setAudioBlob(blob)
        setAudioURL(url)
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)

      toast.success('Recording started')
    } catch (error) {
      console.error('Error accessing microphone:', error)
      toast.error('Failed to access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      toast.success('Recording stopped')
    }
  }

  const resetRecording = () => {
    setAudioBlob(null)
    setAudioURL(null)
    setRecordingTime(0)
    if (timerRef.current) {
      clearInterval(timerRef.current)
    }
  }

  const handleTranscribe = async () => {
    if (!audioBlob) {
      toast.error('No audio recorded. Please record audio first.')
      return
    }

    // Validate blob size
    if (audioBlob.size === 0) {
      toast.error('Recorded audio is empty. Please record again.')
      return
    }

    try {
      // Convert webm to wav (simplified - in production, use proper conversion)
      const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
      await onTranscribe(audioFile)
    } catch (error) {
      console.error('Transcription error:', error)
      toast.error(error.message || 'Failed to transcribe audio. Please try again.')
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Record Audio</h2>
        <p className="text-white/70">Record your voice to transcribe</p>
      </div>

      {/* Recording Controls */}
      <div className="flex flex-col items-center space-y-6">
        {/* Recording Button */}
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={disabled && !isRecording}
          className={`relative w-32 h-32 rounded-full flex items-center justify-center transition-all transform hover:scale-105 ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600 animate-pulse'
              : 'bg-white hover:bg-gray-100'
          } ${disabled && !isRecording ? 'opacity-50 cursor-not-allowed' : 'shadow-2xl'}`}
        >
          {isRecording ? (
            <Square className="w-12 h-12 text-white" />
          ) : (
            <Mic className="w-12 h-12 text-purple-600" />
          )}
        </button>

        {/* Recording Time */}
        {isRecording && (
          <div className="text-white text-2xl font-mono font-bold">
            {formatTime(recordingTime)}
          </div>
        )}

        {/* Recorded Audio Preview */}
        {audioURL && (
          <div className="w-full space-y-4">
            <audio src={audioURL} controls className="w-full" />
            <div className="flex gap-3">
              <button
                onClick={handleTranscribe}
                disabled={disabled}
                className="flex-1 py-3 px-6 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-xl transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Transcribe Audio
              </button>
              <button
                onClick={resetRecording}
                disabled={disabled}
                className="py-3 px-6 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!isRecording && !audioURL && (
          <div className="text-center text-white/70 text-sm max-w-md">
            <p>Click the microphone button to start recording.</p>
            <p className="mt-2">Make sure your microphone permissions are enabled.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default AudioRecorder

