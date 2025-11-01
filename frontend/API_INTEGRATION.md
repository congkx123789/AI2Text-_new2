# Frontend API Integration

This document describes how the frontend integrates with the backend API.

> **Important**: This document only describes the **public API interface**. It does not contain backend implementation details, internal structure, or code specifics.

---

## API Configuration

### Environment Variables

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

Change this to your API server URL in production.

---

## API Endpoints Used

### Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Check if API is available
- **Response**: `{ status: "healthy", timestamp: number }`

### List Models
- **Endpoint**: `GET /models`
- **Purpose**: Get available model names
- **Response**: `{ models: [{ name: string, available: boolean }] }`

### Transcribe Audio
- **Endpoint**: `POST /transcribe`
- **Purpose**: Transcribe audio file
- **Request**: Multipart form data with audio file and settings
- **Response**: `{ text: string, confidence: number, processing_time: number }`

---

## Request/Response Formats

All API calls use standard HTTP methods and JSON responses. See the public API documentation for detailed endpoint specifications.

---

## Error Handling

The frontend handles the following error scenarios:

1. **Network Errors** - Connection failures
2. **HTTP Errors** - 4xx, 5xx status codes
3. **Validation Errors** - Invalid input
4. **Timeout Errors** - Request timeouts

Error messages are user-friendly and do not expose internal details.

---

## Security

- All API calls use HTTPS in production
- CORS is configured on the backend
- No authentication tokens are stored in frontend code
- Sensitive data is not logged

---

## Implementation

The frontend uses Axios for API calls. See `src/services/api.js` for the API client implementation.

---

**Note**: This document describes only the public API interface. For backend details, refer to backend documentation (not included in frontend).

