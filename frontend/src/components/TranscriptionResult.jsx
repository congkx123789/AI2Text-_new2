import { Copy, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'react-hot-toast'

const TranscriptionResult = ({ transcription, isTranscribing }) => {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    if (transcription?.text) {
      navigator.clipboard.writeText(transcription.text)
      setCopied(true)
      toast.success('Copied to clipboard!')
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (isTranscribing) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8">
        <div className="flex flex-col items-center justify-center space-y-4 py-12">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full animate-spin"></div>
          </div>
          <p className="text-white text-lg font-medium">Transcribing...</p>
          <p className="text-white/60 text-sm">This may take a few moments</p>
        </div>
      </div>
    )
  }

  if (!transcription) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8">
        <div className="text-center py-12">
          <div className="w-24 h-24 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-12 h-12 text-white/50" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No Transcription Yet</h3>
          <p className="text-white/60">
            Upload or record audio to see transcription results here
          </p>
        </div>
      </div>
    )
  }

  if (transcription.error) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center">
              <AlertCircle className="w-6 h-6 text-red-400" />
            </div>
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-semibold text-white mb-2">Transcription Failed</h3>
            <p className="text-white/80">{transcription.error}</p>
          </div>
        </div>
      </div>
    )
  }

  const confidencePercentage = (transcription.confidence * 100).toFixed(1)
  const confidenceColor = 
    transcription.confidence >= 0.8 ? 'text-green-400' :
    transcription.confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Transcription Result</h2>
        <button
          onClick={copyToClipboard}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <Copy className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Transcription Text */}
      <div className="bg-white/5 rounded-xl p-6 min-h-[200px]">
        <p className="text-white text-lg leading-relaxed whitespace-pre-wrap">
          {transcription.text || 'No transcription available'}
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 gap-4">
        {/* Confidence Score */}
        <div className="bg-white/5 rounded-xl p-4">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="w-5 h-5 text-white/70" />
            <span className="text-white/70 text-sm font-medium">Confidence</span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className={`text-3xl font-bold ${confidenceColor}`}>
              {confidencePercentage}%
            </span>
          </div>
          {/* Confidence Bar */}
          <div className="mt-3 h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                transcription.confidence >= 0.8 ? 'bg-green-400' :
                transcription.confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
              }`}
              style={{ width: `${confidencePercentage}%` }}
            />
          </div>
        </div>

        {/* Processing Time */}
        <div className="bg-white/5 rounded-xl p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-5 h-5 text-white/70" />
            <span className="text-white/70 text-sm font-medium">Processing</span>
          </div>
          <p className="text-white text-2xl font-bold">
            {transcription.processing_time?.toFixed(2) || 'N/A'}s
          </p>
        </div>
      </div>

      {/* Metadata */}
      <div className="border-t border-white/10 pt-4">
        <div className="text-xs text-white/50 space-y-1">
          {/* Model name removed for security */}
          <div className="flex justify-between">
            <span>Confidence Score:</span>
            <span className="font-mono">{transcription.confidence?.toFixed(4) || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span>Processing Time:</span>
            <span className="font-mono">{transcription.processing_time?.toFixed(3) || 'N/A'}s</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TranscriptionResult

