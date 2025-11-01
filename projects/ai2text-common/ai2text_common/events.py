"""
CloudEvents and NATS message helpers
"""

from typing import Any, Dict, Optional
from datetime import datetime
import json
import uuid

from cloudevents.http import CloudEvent
from cloudevents.conversion import to_json


class CloudEventsHelper:
    """Helper class for creating and parsing CloudEvents"""

    @staticmethod
    def create(
        type: str,
        source: str,
        data: Any,
        id: Optional[str] = None,
        time: Optional[datetime] = None,
    ) -> CloudEvent:
        """
        Create a CloudEvent

        Args:
            type: Event type (e.g., "recording.ingested.v1")
            source: Event source (e.g., "ingestion-service")
            data: Event payload (Pydantic model or dict)
            id: Optional event ID (auto-generated if None)
            time: Optional event timestamp (current time if None)

        Returns:
            CloudEvent instance
        """
        if isinstance(data, dict):
            data_dict = data
        else:
            # Assume Pydantic model
            data_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()

        return CloudEvent(
            {
                "type": type,
                "source": source,
                "id": id or str(uuid.uuid4()),
                "time": (time or datetime.utcnow()).isoformat() + "Z",
                "datacontenttype": "application/json",
            },
            data_dict,
        )

    @staticmethod
    def to_json(event: CloudEvent) -> bytes:
        """Convert CloudEvent to JSON bytes"""
        return to_json(event).encode("utf-8")

    @staticmethod
    def from_json(data: bytes) -> CloudEvent:
        """Parse CloudEvent from JSON bytes"""
        from cloudevents.conversion import from_json

        return from_json(data)

    @staticmethod
    def parse_data(event: CloudEvent, model_class: type) -> Any:
        """
        Parse CloudEvent data into a Pydantic model

        Args:
            event: CloudEvent instance
            model_class: Pydantic model class

        Returns:
            Instance of model_class
        """
        data = event.data
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))
        elif isinstance(data, str):
            data = json.loads(data)

        return model_class(**data)


class NATSHelper:
    """Helper class for NATS message handling"""

    @staticmethod
    async def publish_event(
        client,
        subject: str,
        event: CloudEvent,
    ) -> None:
        """
        Publish a CloudEvent to NATS

        Args:
            client: NATS client instance
            subject: NATS subject
            event: CloudEvent to publish
        """
        data = CloudEventsHelper.to_json(event)
        await client.publish(subject, data)

    @staticmethod
    async def subscribe_events(
        client,
        subject: str,
        handler: callable,
        queue: Optional[str] = None,
    ):
        """
        Subscribe to CloudEvents on NATS

        Args:
            client: NATS client instance
            subject: NATS subject pattern
            handler: Async function that receives (event: CloudEvent) -> None
            queue: Optional queue group name
        """
        async def message_handler(msg):
            event = CloudEventsHelper.from_json(msg.data)
            await handler(event)

        sub = await client.subscribe(subject, cb=message_handler, queue=queue)
        return sub


