# Public API Documentation

This document describes the public API endpoints for the Vietnamese ASR system.

> **Note**: This documentation does not expose internal implementation details, file structures, or backend code. It only describes the public API interface.

---

## Base URL

```
http://localhost:8000
```

For production, use your deployed API URL.

---

## Endpoints

### `GET /`

**Description**: API information

**Response**:
```json
{
  "service": "Vietnamese ASR API",
  "version": "1.0.0",
  "endpoints": {
    "transcribe": "POST /transcribe",
    "models": "GET /models",
    "health": "GET /health"
  }
}
```

---

### `GET /health`

**Description**: Check API health status

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1234567890
}
```

**Status Codes**:
- `200 OK` - API is healthy
- `503 Service Unavailable` - API is not available

---

### `GET /models`

**Description**: List available ASR models

**Response**:
```json
{
  "models": [
    {
      "name": "default",
      "available": true
    },
    {
      "name": "model_v2",
      "available": true
    }
  ]
}
```

**Note**: This endpoint only returns model names and availability. Internal paths and details are not exposed.

---

### `POST /transcribe`

**Description**: Transcribe audio file to text

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Fields**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | File | Yes | - | Audio file (WAV, MP3, FLAC, WebM) |
| `model_name` | String | No | "default" | Model name to use |
| `use_beam_search` | Boolean | No | true | Enable beam search decoding |
| `beam_width` | Integer | No | 5 | Beam search width (3-20) |
| `use_lm` | Boolean | No | false | Use language model (if available) |
| `min_confidence` | Float | No | 0.5 | Minimum confidence threshold (0-1) |

**Response**:
```json
{
  "text": "xin chào việt nam",
  "confidence": 0.95,
  "processing_time": 1.23
}
```

**Error Response**:
```json
{
  "detail": "Error message (user-friendly)"
}
```

**Status Codes**:
- `200 OK` - Transcription successful
- `400 Bad Request` - Invalid input
- `413 Payload Too Large` - File too large (>50MB)
- `415 Unsupported Media Type` - Invalid file type
- `500 Internal Server Error` - Server error

---

## File Requirements

### Supported Formats
- WAV
- MP3
- FLAC
- WebM

### File Size Limit
- Maximum: 50MB

---

## Error Handling

All errors return a standardized format:

```json
{
  "detail": "Human-readable error message"
}
```

**Common Errors**:
- `"File too large. Maximum file size is 50MB."`
- `"Unsupported file type. Please use WAV, MP3, FLAC, or WebM."`
- `"Transcription failed. Please try again."`
- `"Network error. Please check your internet connection."`

---

## Security

### Headers
The API includes security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`

### CORS
CORS is configured for specific frontend origins. Contact your API administrator for CORS configuration.

---

## Rate Limiting

Rate limiting may be applied. Check response headers for rate limit information.

---

## Authentication

Authentication may be required in production. Check with your API administrator for authentication requirements.

---

## Example Usage

### Using cURL

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@test.wav" \
  -F "use_beam_search=true" \
  -F "beam_width=5"
```

### Using JavaScript (Fetch)

```javascript
const formData = new FormData();
formData.append('audio', audioFile);
formData.append('use_beam_search', 'true');
formData.append('beam_width', '5');

const response = await fetch('http://localhost:8000/transcribe', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.text);
```

---

## Notes

- This is public API documentation only
- Internal implementation details are not exposed
- For backend development, refer to internal documentation
- API endpoints may change - check version for updates

---

## Support

For API support or issues, contact your API administrator.

