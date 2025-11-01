/**
 * Validation utilities for frontend
 */

/**
 * Validate audio file
 * @param {File} file - File to validate
 * @returns {Object} Validation result { valid: boolean, error?: string }
 */
export const validateAudioFile = (file) => {
  if (!file) {
    return { valid: false, error: 'No file provided' }
  }

  // Check if it's a File object
  if (!(file instanceof File)) {
    return { valid: false, error: 'Invalid file object' }
  }

  // Validate file type
  const validTypes = [
    'audio/wav',
    'audio/mpeg',
    'audio/mp3',
    'audio/flac',
    'audio/webm',
    'audio/x-wav',
    'audio/wave',
    'audio/mp4',
    'audio/m4a'
  ]
  
  const validExtensions = ['.wav', '.mp3', '.flac', '.webm', '.m4a', '.wma', '.aac']
  const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
  
  const isValidType = validTypes.includes(file.type)
  const isValidExtension = validExtensions.some(ext => 
    file.name.toLowerCase().endsWith(ext)
  )

  if (!isValidType && !isValidExtension) {
    return { 
      valid: false, 
      error: `Invalid file type. Supported formats: ${validExtensions.join(', ')}` 
    }
  }

  // Validate file size (50MB max)
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

/**
 * Validate transcription settings
 * @param {Object} settings - Settings to validate
 * @returns {Object} Validation result { valid: boolean, errors?: string[] }
 */
export const validateSettings = (settings) => {
  const errors = []

  // Validate model name
  if (!settings.modelName || typeof settings.modelName !== 'string') {
    errors.push('Model name is required')
  }

  // Validate beam width
  if (settings.useBeamSearch) {
    const beamWidth = Number(settings.beamWidth)
    if (isNaN(beamWidth) || beamWidth < 1 || beamWidth > 100) {
      errors.push('Beam width must be between 1 and 100')
    }
  }

  // Validate confidence threshold
  const minConfidence = Number(settings.minConfidence)
  if (isNaN(minConfidence) || minConfidence < 0 || minConfidence > 1) {
    errors.push('Confidence threshold must be between 0 and 1')
  }

  // Validate LM path if LM is enabled
  if (settings.useLm && settings.lmPath) {
    if (typeof settings.lmPath !== 'string' || settings.lmPath.trim() === '') {
      errors.push('Language model path is invalid')
    }
  }

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  }
}

/**
 * Sanitize filename
 * @param {string} filename - Filename to sanitize
 * @returns {string} Sanitized filename
 */
export const sanitizeFilename = (filename) => {
  // Remove path components
  filename = filename.replace(/^.*[\\\/]/, '')
  
  // Remove or replace invalid characters
  filename = filename.replace(/[<>:"|?*]/g, '_')
  
  // Limit length
  if (filename.length > 255) {
    const ext = filename.substring(filename.lastIndexOf('.'))
    filename = filename.substring(0, 255 - ext.length) + ext
  }
  
  return filename
}

export default {
  validateAudioFile,
  validateSettings,
  sanitizeFilename,
}

