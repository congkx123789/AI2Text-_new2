import { useState, useEffect } from 'react'
import { checkHealth } from '../services/api'

/**
 * Custom hook to check API health periodically
 * @param {number} interval - Check interval in milliseconds (default: 30000 = 30 seconds)
 * @returns {Object} API health status
 */
export const useApiHealth = (interval = 30000) => {
  const [isHealthy, setIsHealthy] = useState(null) // null = checking, true = healthy, false = unhealthy
  const [isChecking, setIsChecking] = useState(true)
  const [error, setError] = useState(null)

  const check = async () => {
    setIsChecking(true)
    try {
      const health = await checkHealth()
      setIsHealthy(true)
      setError(null)
    } catch (err) {
      setIsHealthy(false)
      setError(err.message)
    } finally {
      setIsChecking(false)
    }
  }

  useEffect(() => {
    // Initial check
    check()

    // Set up periodic checks
    const healthInterval = setInterval(check, interval)

    return () => {
      clearInterval(healthInterval)
    }
  }, [interval])

  return {
    isHealthy,
    isChecking,
    error,
    check, // Manual refresh function
  }
}

export default useApiHealth

