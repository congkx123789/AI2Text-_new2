"""Validators for OpenAPI/AsyncAPI contract compliance."""
import json
from typing import Dict, Any
import jsonschema
import yaml


def load_openapi_spec(service: str) -> Dict[str, Any]:
    """Load OpenAPI spec for service."""
    spec_path = f"../../../ai2text-contracts/openapi/{service}.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


def load_asyncapi_spec(event: str) -> Dict[str, Any]:
    """Load AsyncAPI spec for event."""
    spec_path = f"../../../ai2text-contracts/asyncapi/{event}.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


def validate_openapi_response(
    response_data: Dict[str, Any],
    service: str,
    endpoint: str,
    method: str = "get",
    status_code: int = 200,
) -> bool:
    """Validate response matches OpenAPI spec."""
    spec = load_openapi_spec(service)
    
    # Find endpoint definition
    path_item = spec["paths"].get(endpoint)
    if not path_item:
        raise ValueError(f"Endpoint {endpoint} not found in spec")
    
    operation = path_item.get(method.lower())
    if not operation:
        raise ValueError(f"Method {method} not found for {endpoint}")
    
    # Get response schema
    responses = operation.get("responses", {})
    response_def = responses.get(str(status_code))
    if not response_def:
        raise ValueError(f"Status {status_code} not defined for {endpoint}")
    
    content = response_def.get("content", {})
    schema_ref = content.get("application/json", {}).get("schema")
    
    if schema_ref and "$ref" in schema_ref:
        # Resolve schema reference
        ref_path = schema_ref["$ref"].replace("#/", "").split("/")
        schema = spec
        for part in ref_path:
            schema = schema[part]
    else:
        schema = schema_ref
    
    if not schema:
        # No schema defined, skip validation
        return True
    
    # Convert OpenAPI schema to JSON Schema
    json_schema = _openapi_to_json_schema(schema)
    
    # Validate
    try:
        jsonschema.validate(instance=response_data, schema=json_schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e.message}")
        print(f"Response data: {json.dumps(response_data, indent=2)}")
        return False


def validate_asyncapi_event(
    event_data: Dict[str, Any],
    event_subject: str,
) -> bool:
    """Validate event matches AsyncAPI spec."""
    # Extract event name from subject (e.g., "transcription.completed.v1")
    event_name = event_subject.rsplit(".", 1)[0]  # Remove version
    
    spec = load_asyncapi_spec(event_name)
    
    # Find message schema
    channels = spec.get("channels", {})
    channel = channels.get(event_subject)
    if not channel:
        # Try without version
        for ch_name, ch_data in channels.items():
            if ch_name.startswith(event_name):
                channel = ch_data
                break
    
    if not channel:
        raise ValueError(f"Event {event_subject} not found in spec")
    
    messages = channel.get("messages", {})
    message = list(messages.values())[0]  # Get first message
    
    payload = message.get("payload", {})
    schema = payload.get("schema") or payload
    
    if not schema:
        return True
    
    # Convert to JSON Schema
    json_schema = _openapi_to_json_schema(schema)
    
    try:
        jsonschema.validate(instance=event_data, schema=json_schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Event validation error: {e.message}")
        print(f"Event data: {json.dumps(event_data, indent=2)}")
        return False


def _openapi_to_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAPI schema to JSON Schema."""
    # Simple conversion - handle common cases
    json_schema = schema.copy()
    
    # Handle type conversions
    if "type" in json_schema:
        if json_schema["type"] == "integer":
            json_schema["type"] = "number"
    
    return json_schema

