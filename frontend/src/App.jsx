import { useState, useEffect } from 'react'
import { Mic, Upload, Settings, CheckCircle, AlertCircle, Loader2, Wifi, WifiOff } from 'lucide-react'
import AudioRecorder from './components/AudioRecorder'
import AudioUploader from './components/AudioUploader'
import TranscriptionResult from './components/TranscriptionResult'
import SettingsPanel from './components/SettingsPanel'
import ErrorBoundary from './components/ErrorBoundary'
import { getModels, transcribeAudio, checkHealth } from './services/api'
import { useApiHealth } from './hooks/useApiHealth'
import { Toaster, toast } from 'react-hot-toast'

function App() {
  const [activeTab, setActiveTab] = useState('record') // 'record' or 'upload'
  const [transcription, setTranscription] = useState(null)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [models, setModels] = useState([])
  const [settings, setSettings] = useState({
    modelName: 'default',
    useBeamSearch: true,
    beamWidth: 5,
    useLm: false,
    lmPath: null,
    minConfidence: 0.5,
    showSettings: false
  })

  // API health check
  const { isHealthy, isChecking, error: apiError } = useApiHealth(30000) // Check every 30 seconds

  useEffect(() => {
    loadModels()
    
    // Initial health check
    checkApiConnection()
  }, [])

  // Show warning if API is not healthy
  useEffect(() => {
    if (isHealthy === false && apiError) {
      toast.error(`API connection failed: ${apiError}`, {
        duration: 5000,
        icon: '⚠️',
      })
    }
  }, [isHealthy, apiError])

  const checkApiConnection = async () => {
    try {
      await checkHealth()
    } catch (error) {
      console.error('API health check failed:', error)
    }
  }

  const loadModels = async () => {
    try {
      const response = await getModels()
      setModels(response.models || [])
      
      // Auto-select first model if available
      if (response.models && response.models.length > 0 && settings.modelName === 'default') {
        setSettings(prev => ({ ...prev, modelName: response.models[0].name }))
      }
    } catch (error) {
      console.error('Failed to load models:', error)
      toast.error(`Failed to load models: ${error.message}`, {
        duration: 4000,
      })
    }
  }

  const handleTranscribe = async (audioFile) => {
    // Validate API connection first
    if (isHealthy === false) {
      toast.error('API server is not available. Please start the backend API first.', {
        duration: 5000,
      })
      return
    }

    setIsTranscribing(true)
    setTranscription(null)
    setUploadProgress(0)

    try {
      const result = await transcribeAudio(
        audioFile,
        settings,
        (progress) => {
          setUploadProgress(progress)
        }
      )
      
      setTranscription(result)
      toast.success('Transcription completed successfully!', {
        duration: 3000,
      })
    } catch (error) {
      console.error('Transcription error:', error)
      
      // Parse error message
      let errorMessage = 'Transcription failed. Please try again.'
      if (error.message) {
        errorMessage = error.message
      }
      
      setTranscription({
        error: errorMessage
      })
      
      toast.error(errorMessage, {
        duration: 5000,
      })
    } finally {
      setIsTranscribing(false)
      setUploadProgress(0)
    }
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-purple-800">
        <Toaster 
          position="top-right" 
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#10b981',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
        
        {/* Header */}
        <header className="bg-white/10 backdrop-blur-md border-b border-white/20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white">Vietnamese ASR</h1>
                <p className="text-white/80 text-sm mt-1">Speech to Text System</p>
              </div>
              <div className="flex items-center space-x-3">
                {/* API Status Indicator */}
                <div className="flex items-center space-x-2">
                  {isChecking ? (
                    <Loader2 className="w-5 h-5 text-white/60 animate-spin" />
                  ) : isHealthy ? (
                    <Wifi className="w-5 h-5 text-green-400" title="API Connected" />
                  ) : (
                    <WifiOff className="w-5 h-5 text-red-400" title="API Disconnected" />
                  )}
                </div>
                <button
                  onClick={() => setSettings({ ...settings, showSettings: !settings.showSettings })}
                  className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
                  title="Settings"
                >
                  <Settings className="w-6 h-6" />
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* API Status Banner */}
        {isHealthy === false && (
          <div className="bg-red-500/90 text-white px-4 py-2 text-center text-sm">
            <AlertCircle className="w-4 h-4 inline-block mr-2" />
            API server is not responding. Please start the backend API at <code className="bg-black/20 px-2 py-1 rounded">python api/app.py</code>
          </div>
        )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Settings Panel */}
        {settings.showSettings && (
          <SettingsPanel
            settings={settings}
            models={models}
            onSettingsChange={(newSettings) => setSettings({ ...settings, ...newSettings })}
            onClose={() => setSettings({ ...settings, showSettings: false })}
          />
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Audio Input */}
          <div className="space-y-6">
            {/* Tab Selection */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-1 flex gap-2">
              <button
                onClick={() => setActiveTab('record')}
                className={`flex-1 py-3 px-4 rounded-xl font-medium transition-all ${
                  activeTab === 'record'
                    ? 'bg-white text-purple-600 shadow-lg'
                    : 'text-white/80 hover:text-white hover:bg-white/5'
                }`}
              >
                <Mic className="w-5 h-5 inline-block mr-2" />
                Record
              </button>
              <button
                onClick={() => setActiveTab('upload')}
                className={`flex-1 py-3 px-4 rounded-xl font-medium transition-all ${
                  activeTab === 'upload'
                    ? 'bg-white text-purple-600 shadow-lg'
                    : 'text-white/80 hover:text-white hover:bg-white/5'
                }`}
              >
                <Upload className="w-5 h-5 inline-block mr-2" />
                Upload
              </button>
            </div>

            {/* Audio Input Component */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8">
              {activeTab === 'record' ? (
                <AudioRecorder onTranscribe={handleTranscribe} disabled={isTranscribing} />
              ) : (
                <AudioUploader onTranscribe={handleTranscribe} disabled={isTranscribing} />
              )}

              {isTranscribing && (
                <div className="mt-6 space-y-3">
                  <div className="flex items-center justify-center space-x-3 text-white">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Transcribing audio...</span>
                  </div>
                  {uploadProgress > 0 && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-white/70 text-sm">
                        <span>Upload progress</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Settings Summary */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6">
              <h3 className="text-white font-semibold mb-4">Current Settings</h3>
              <div className="space-y-2 text-sm text-white/90">
                <div className="flex justify-between">
                  <span>Model:</span>
                  <span className="font-mono">{settings.modelName}</span>
                </div>
                <div className="flex justify-between">
                  <span>Beam Search:</span>
                  <span className="font-mono">{settings.useBeamSearch ? `Yes (w=${settings.beamWidth})` : 'No'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Language Model:</span>
                  <span className="font-mono">{settings.useLm ? 'Enabled' : 'Disabled'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Min Confidence:</span>
                  <span className="font-mono">{settings.minConfidence.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Results */}
          <div>
            <TranscriptionResult
              transcription={transcription}
              isTranscribing={isTranscribing}
            />
          </div>
        </div>
      </main>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center text-white/60 text-sm">
          <p>Vietnamese ASR System - Powered by Deep Learning</p>
          {isHealthy === false && (
            <p className="mt-2 text-red-300 text-xs">
              ⚠️ API Server Offline - Start backend with: python api/app.py
            </p>
          )}
        </footer>
      </div>
    </ErrorBoundary>
  )
}

export default App

