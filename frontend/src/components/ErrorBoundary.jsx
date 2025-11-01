import React from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true }
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    this.setState({
      error,
      errorInfo
    })
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-red-600 via-red-700 to-red-800 flex items-center justify-center p-4">
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 max-w-2xl w-full">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center">
                  <AlertCircle className="w-8 h-8 text-red-400" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-white mb-2">Something went wrong</h2>
                <p className="text-white/90 mb-4">
                  An unexpected error occurred. Please try refreshing the page.
                </p>
                
                {this.state.error && (
                  <details className="mt-4 bg-black/20 rounded-lg p-4">
                    <summary className="text-white/80 cursor-pointer text-sm font-medium mb-2">
                      Error Details
                    </summary>
                    <pre className="text-xs text-white/60 overflow-auto max-h-40 font-mono">
                      {this.state.error.toString()}
                      {this.state.errorInfo?.componentStack}
                    </pre>
                  </details>
                )}
                
                <button
                  onClick={this.handleReset}
                  className="mt-6 px-6 py-3 bg-white hover:bg-gray-100 text-red-600 font-semibold rounded-xl transition-all flex items-center space-x-2"
                >
                  <RefreshCw className="w-5 h-5" />
                  <span>Refresh Page</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary

