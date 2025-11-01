import axios from 'axios'
import { toast } from 'react-hot-toast'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 120 seconds for transcription (longer for large files)
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor - Add common headers, logging
api.interceptors.request.use(
  (config) => {
    // Add request timestamp for tracking
    config.metadata = { startTime: new Date() }
    
    // Log request (only in development)
    if (import.meta.env.DEV) {
      console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`, config.params || config.data)
    }
    
    return config
  },
  (error) => {
    console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

// Response interceptor - Handle errors globally
api.interceptors.response.use(
  (response) => {
    // Calculate request duration
    if (response.config.metadata?.startTime) {
      const duration = new Date() - response.config.metadata.startTime
      if (import.meta.env.DEV) {
        console.log(`[API Response] ${response.config.url} - ${duration}ms`)
      }
    }
    
    return response
  },
  async (error) => {
    const originalRequest = error.config
    
    // Handle network errors
    if (!error.response) {
      if (error.code === 'ECONNABORTED') {
        // Timeout error
        error.message = 'Request timeout. The server took too long to respond.'
      } else if (error.message === 'Network Error') {
        // Network connection error
        error.message = 'Network error. Please check your internet connection and ensure the API server is running.'
      } else {
        error.message = error.message || 'Network error occurred'
      }
      
      // Don't retry on network errors for non-GET requests
      if (originalRequest && originalRequest.method !== 'get') {
        return Promise.reject(error)
      }
    }
    
    // Handle HTTP errors
    if (error.response) {
      const status = error.response.status
      const data = error.response.data
      
      switch (status) {
        case 400:
          error.message = data?.detail || data?.message || 'Bad request. Please check your input.'
          break
        case 401:
          error.message = 'Unauthorized. Please check your credentials.'
          break
        case 403:
          error.message = 'Forbidden. You do not have permission to access this resource.'
          break
        case 404:
          error.message = data?.detail || 'Resource not found.'
          break
        case 413:
          error.message = 'File too large. Maximum file size is 50MB.'
          break
        case 415:
          error.message = 'Unsupported file type. Please use WAV, MP3, FLAC, or WebM.'
          break
        case 422:
          error.message = data?.detail || data?.message || 'Validation error. Please check your input.'
          break
        case 429:
          error.message = 'Too many requests. Please try again later.'
          break
        case 500:
          error.message = data?.detail || 'Internal server error. Please try again later.'
          break
        case 503:
          error.message = 'Service unavailable. The server is temporarily down.'
          break
        default:
          error.message = data?.detail || data?.message || `Error ${status}: ${error.message}`
      }
    }
    
    // Retry logic for certain errors (only for GET requests)
    if (originalRequest && !originalRequest._retry && error.response?.status >= 500) {
      if (originalRequest.method === 'get') {
        originalRequest._retry = true
        await new Promise(resolve => setTimeout(resolve, 1000)) // Wait 1 second
        return api(originalRequest)
      }
    }
    
    return Promise.reject(error)
  }
)

// Error handler utility
const handleApiError = (error, defaultMessage = 'An error occurred') => {
  let message = defaultMessage
  
  if (error.response) {
    // Server responded with error
    message = error.message || error.response.data?.detail || error.response.data?.message || defaultMessage
  } else if (error.request) {
    // Request made but no response
    message = 'No response from server. Please check if the API server is running.'
  } else {
    // Error in request setup
    message = error.message || defaultMessage
  }
  
  return message
}

// API Functions with comprehensive error handling

/**
 * Get list of available models
 * @returns {Promise<Object>} Models list
 */
export const getModels = async () => {
  try {
    const response = await api.get('/models')
    return response.data
  } catch (error) {
    const message = handleApiError(error, 'Failed to load models')
    console.error('[getModels Error]', error)
    throw new Error(message)
  }
}

/**
 * Transcribe audio file
 * @param {File} audioFile - Audio file to transcribe
 * @param {Object} settings - Transcription settings
 * @param {string} settings.modelName - Model name
 * @param {boolean} settings.useBeamSearch - Use beam search
 * @param {number} settings.beamWidth - Beam width
 * @param {boolean} settings.useLm - Use language model
 * @param {string} settings.lmPath - Language model path
 * @param {number} settings.minConfidence - Minimum confidence threshold
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<Object>} Transcription result
 */
export const transcribeAudio = async (audioFile, settings, onProgress = null) => {
  // Validation
  if (!audioFile) {
    throw new Error('No audio file provided')
  }
  
  // Validate file type
  const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/webm', 'audio/x-wav', 'audio/wave']
  const validExtensions = ['.wav', '.mp3', '.flac', '.webm', '.m4a']
  const fileExtension = audioFile.name.substring(audioFile.name.lastIndexOf('.')).toLowerCase()
  
  if (!validTypes.includes(audioFile.type) && !validExtensions.includes(fileExtension)) {
    throw new Error('Invalid file type. Please use WAV, MP3, FLAC, or WebM format.')
  }
  
  // Validate file size (50MB max)
  const maxSize = 50 * 1024 * 1024 // 50MB
  if (audioFile.size > maxSize) {
    throw new Error(`File size exceeds maximum of 50MB. Current size: ${(audioFile.size / 1024 / 1024).toFixed(2)}MB`)
  }
  
  // Build form data
  const formData = new FormData()
  formData.append('audio', audioFile)
  formData.append('model_name', settings.modelName || 'default')
  formData.append('use_beam_search', settings.useBeamSearch ? 'true' : 'false')
  formData.append('beam_width', String(settings.beamWidth || 5))
  formData.append('use_lm', settings.useLm ? 'true' : 'false')
  
  if (settings.lmPath) {
    formData.append('lm_path', settings.lmPath)
  }
  
  if (settings.minConfidence !== undefined && settings.minConfidence !== null) {
    formData.append('min_confidence', String(settings.minConfidence))
  }
  
  try {
    const response = await api.post('/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes for large files
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(percentCompleted)
        }
      },
    })
    
    // Validate response
    if (!response.data) {
      throw new Error('Empty response from server')
    }
    
    return response.data
  } catch (error) {
    const message = handleApiError(error, 'Transcription failed')
    console.error('[transcribeAudio Error]', error)
    throw new Error(message)
  }
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health', {
      timeout: 5000, // Short timeout for health check
    })
    return response.data
  } catch (error) {
    const message = handleApiError(error, 'API server is not responding')
    throw new Error(message)
  }
}

/**
 * Check if API is reachable
 * @returns {Promise<boolean>} True if API is reachable
 */
export const isApiReachable = async () => {
  try {
    await checkHealth()
    return true
  } catch (error) {
    return false
  }
}

export default api

