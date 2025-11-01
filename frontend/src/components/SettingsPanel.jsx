import { X } from 'lucide-react'

const SettingsPanel = ({ settings, models, onSettingsChange, onClose }) => {
  const handleChange = (key, value) => {
    onSettingsChange({ [key]: value })
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <X className="w-6 h-6 text-gray-600" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model
            </label>
            <select
              value={settings.modelName}
              onChange={(e) => handleChange('modelName', e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="default">Default</option>
              {models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name} ({model.size_mb?.toFixed(1)}MB)
                </option>
              ))}
            </select>
          </div>

          {/* Beam Search */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Beam Search
              </label>
              <input
                type="checkbox"
                checked={settings.useBeamSearch}
                onChange={(e) => handleChange('useBeamSearch', e.target.checked)}
                className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
              />
            </div>
            {settings.useBeamSearch && (
              <div className="mt-2">
                <label className="block text-xs text-gray-600 mb-1">
                  Beam Width: {settings.beamWidth}
                </label>
                <input
                  type="range"
                  min="3"
                  max="20"
                  value={settings.beamWidth}
                  onChange={(e) => handleChange('beamWidth', parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            )}
          </div>

          {/* Language Model */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Use Language Model (KenLM)
              </label>
              <input
                type="checkbox"
                checked={settings.useLm}
                onChange={(e) => handleChange('useLm', e.target.checked)}
                className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
              />
            </div>
            {settings.useLm && (
              <div className="mt-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LM Path (optional)
                </label>
                <input
                  type="text"
                  value={settings.lmPath || ''}
                  onChange={(e) => handleChange('lmPath', e.target.value)}
                  placeholder="models/lm.arpa"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Path to KenLM .arpa file. Leave empty to use default.
                </p>
              </div>
            )}
          </div>

          {/* Minimum Confidence */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Minimum Confidence: {settings.minConfidence.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={settings.minConfidence}
              onChange={(e) => handleChange('minConfidence', parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Predictions below this threshold will be rejected.
            </p>
          </div>

          {/* Info Section */}
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-purple-900 mb-2">Tips:</h3>
            <ul className="text-sm text-purple-800 space-y-1">
              <li>• Beam Search improves accuracy but is slower</li>
              <li>• Language Model provides 10-30% WER improvement</li>
              <li>• Higher beam width = better accuracy, slower processing</li>
              <li>• Confidence threshold filters low-quality predictions</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 p-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel

