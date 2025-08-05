import json
import re
from typing import Any
from datetime import datetime
from enum import Enum

import pytest

from symai.models.base import (
    LLMDataModel,
    build_dynamic_llm_datamodel,
)


# ---------------------------------------------------------------------------
# Real-world Model Examples
# ---------------------------------------------------------------------------
class APIResponse(LLMDataModel):
    status_code: int
    headers: dict[str, str]
    body: dict[str, Any]
    timestamp: str
    request_id: str | None = None


class UserProfile(LLMDataModel):
    user_id: str
    username: str
    email: str
    full_name: str | None
    bio: str | None
    avatar_url: str | None
    verified: bool = False
    created_at: str
    settings: 'UserSettings'
    social_links: list['SocialLink']
    metadata: dict[str, Any]


class UserSettings(LLMDataModel):
    theme: str = "light"
    language: str = "en"
    notifications_enabled: bool = True
    privacy_level: str = "public"
    timezone: str = "UTC"


class SocialLink(LLMDataModel):
    platform: str
    url: str
    verified: bool = False


class DatabaseConfig(LLMDataModel):
    host: str
    port: int
    database: str
    username: str
    password: str | None = None
    ssl_enabled: bool = False
    pool_size: int = 10
    timeout: int = 30
    options: dict[str, str | int | bool] = None


class EventLog(LLMDataModel):
    event_id: str
    event_type: str
    timestamp: str
    user_id: str | None
    session_id: str | None
    payload: dict[str, Any]
    metadata: dict[str, str | int | float | bool]
    tags: list[str]
    severity: str = "info"


class MLModelConfig(LLMDataModel):
    model_name: str
    version: str
    architecture: str
    hyperparameters: dict[str, float | int | str]
    training_config: 'TrainingConfig'
    inference_config: 'InferenceConfig'
    metrics: dict[str, float]


class TrainingConfig(LLMDataModel):
    batch_size: int
    learning_rate: float
    epochs: int
    optimizer: str
    loss_function: str
    early_stopping: bool = True
    checkpoint_interval: int = 10


class InferenceConfig(LLMDataModel):
    batch_size: int = 1
    max_sequence_length: int = 512
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


class GraphQLQuery(LLMDataModel):
    query: str
    variables: dict[str, Any] | None = None
    operation_name: str | None = None


class GraphQLResponse(LLMDataModel):
    data: dict[str, Any] | None
    errors: list[dict[str, Any]] | None = None
    extensions: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Integration Tests with Complex Scenarios
# ---------------------------------------------------------------------------
def test_api_response_with_nested_data():
    """Test realistic API response modeling."""
    response = APIResponse(
        status_code=200,
        headers={"Content-Type": "application/json", "X-Request-ID": "abc123"},
        body={
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "total": 2,
            "page": 1
        },
        timestamp="2024-01-01T00:00:00Z",
        request_id="req_123"
    )

    s = str(response)
    assert "status_code: 200" in s
    assert "Content-Type: application/json" in s
    assert "users:" in s

    schema = APIResponse.simplify_json_schema()
    assert "status_code" in schema
    assert "integer" in schema


def test_user_profile_complete_workflow():
    """Test complete user profile creation and serialization."""
    profile = UserProfile(
        user_id="user_123",
        username="johndoe",
        email="john@example.com",
        full_name="John Doe",
        bio="Software engineer and open source enthusiast",
        avatar_url="https://example.com/avatar.jpg",
        verified=True,
        created_at="2023-01-01T00:00:00Z",
        settings=UserSettings(
            theme="dark",
            language="en",
            notifications_enabled=True,
            privacy_level="friends",
            timezone="America/New_York"
        ),
        social_links=[
            SocialLink(platform="github", url="https://github.com/johndoe", verified=True),
            SocialLink(platform="twitter", url="https://twitter.com/johndoe", verified=False)
        ],
        metadata={"last_login": "2024-01-01", "posts_count": 42}
    )

    s = str(profile)
    assert "johndoe" in s
    assert "theme: dark" in s
    assert "platform: github" in s

    instr = UserProfile.instruct_llm()
    assert "[[Result]]" in instr
    assert "user_id" in instr


def test_database_config_with_validation():
    """Test database configuration with custom validation."""
    class ValidatedDatabaseConfig(DatabaseConfig):
        def validate(self) -> str | None:
            if self.port < 1 or self.port > 65535:
                return f"Invalid port number: {self.port}"
            if self.pool_size < 1:
                return f"Pool size must be positive: {self.pool_size}"
            if self.timeout < 1:
                return f"Timeout must be positive: {self.timeout}"
            return None

        def remedy(self) -> str | None:
            if self.port < 1 or self.port > 65535:
                return "Use a port number between 1 and 65535"
            if self.pool_size < 1:
                return "Set pool_size to at least 1"
            if self.timeout < 1:
                return "Set timeout to at least 1 second"
            return None

    config = ValidatedDatabaseConfig(
        host="localhost",
        port=5432,
        database="myapp",
        username="dbuser",
        ssl_enabled=True,
        pool_size=20,
        timeout=60
    )

    assert config.validate() is None
    assert config.remedy() is None

    invalid_config = ValidatedDatabaseConfig(
        host="localhost",
        port=70000,
        database="myapp",
        username="dbuser",
        pool_size=0,
        timeout=0
    )

    validation_error = invalid_config.validate()
    assert "Invalid port number" in validation_error

    remedy_suggestion = invalid_config.remedy()
    assert "port number between" in remedy_suggestion


def test_event_log_aggregation():
    """Test event log aggregation scenario."""
    events = [
        EventLog(
            event_id=f"evt_{i}",
            event_type="user_action" if i % 2 == 0 else "system_event",
            timestamp=f"2024-01-01T{i:02d}:00:00Z",
            user_id=f"user_{i % 3}" if i % 2 == 0 else None,
            session_id=f"session_{i // 5}",
            payload={"action": f"action_{i}", "details": {"count": i}},
            metadata={"latency_ms": i * 10.5, "success": i % 3 != 0},
            tags=[f"tag_{i % 2}", f"category_{i % 3}"],
            severity="info" if i < 5 else "warning"
        )
        for i in range(10)
    ]

    for event in events[:3]:
        s = str(event)
        assert "event_id:" in s
        assert "event_type:" in s
        assert "payload:" in s


def test_ml_model_config_serialization():
    """Test ML model configuration with nested configs."""
    ml_config = MLModelConfig(
        model_name="text-classifier-v2",
        version="2.1.0",
        architecture="transformer",
        hyperparameters={
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1,
            "activation": "gelu"
        },
        training_config=TrainingConfig(
            batch_size=32,
            learning_rate=0.0001,
            epochs=100,
            optimizer="adamw",
            loss_function="cross_entropy",
            early_stopping=True,
            checkpoint_interval=5
        ),
        inference_config=InferenceConfig(
            batch_size=8,
            max_sequence_length=256,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        ),
        metrics={
            "accuracy": 0.94,
            "precision": 0.92,
            "recall": 0.93,
            "f1_score": 0.925
        }
    )

    s = str(ml_config)
    assert "model_name: text-classifier-v2" in s
    assert "learning_rate: 0.0001" in s
    assert "temperature: 0.7" in s
    assert "accuracy: 0.94" in s

    example = LLMDataModel.generate_example_json(MLModelConfig)
    assert "hyperparameters" in example
    assert "training_config" in example
    assert "inference_config" in example


def test_graphql_query_response_cycle():
    """Test GraphQL query and response modeling."""
    query = GraphQLQuery(
        query="""
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                name
                email
                posts {
                    id
                    title
                }
            }
        }
        """,
        variables={"id": "123"},
        operation_name="GetUser"
    )

    response = GraphQLResponse(
        data={
            "user": {
                "id": "123",
                "name": "Alice",
                "email": "alice@example.com",
                "posts": [
                    {"id": "1", "title": "First Post"},
                    {"id": "2", "title": "Second Post"}
                ]
            }
        },
        errors=None,
        extensions={"tracing": {"duration": 45}}
    )

    query_str = str(query)
    assert "query:" in query_str
    assert "variables:" in query_str

    response_str = str(response)
    assert "Alice" in response_str
    assert "First Post" in response_str


def test_dynamic_api_response_parsing():
    """Test dynamic model building for varying API responses."""
    response_types = [
        list[dict[str, str | int]],
        dict[str, list[str]],
        dict[str, dict[str, Any]],
        list[str] | dict[str, str] | None
    ]

    for response_type in response_types:
        DynamicResponse = build_dynamic_llm_datamodel(response_type)
        example = LLMDataModel.generate_example_json(DynamicResponse)
        assert "value" in example

        instr = DynamicResponse.instruct_llm()
        assert "[[Result]]" in instr


def test_complex_nested_configuration():
    """Test deeply nested configuration structures."""
    class ServiceConfig(LLMDataModel):
        name: str
        endpoints: list['EndpointConfig']
        dependencies: dict[str, 'DependencyConfig']
        health_checks: list['HealthCheckConfig']

    class EndpointConfig(LLMDataModel):
        path: str
        methods: list[str]
        auth_required: bool
        rate_limit: int | None

    class DependencyConfig(LLMDataModel):
        url: str
        timeout: int
        retry_count: int
        circuit_breaker_enabled: bool

    class HealthCheckConfig(LLMDataModel):
        name: str
        endpoint: str
        interval: int
        timeout: int
        failure_threshold: int

    service = ServiceConfig(
        name="user-service",
        endpoints=[
            EndpointConfig(
                path="/users",
                methods=["GET", "POST"],
                auth_required=True,
                rate_limit=100
            ),
            EndpointConfig(
                path="/users/{id}",
                methods=["GET", "PUT", "DELETE"],
                auth_required=True,
                rate_limit=50
            )
        ],
        dependencies={
            "auth": DependencyConfig(
                url="http://auth-service",
                timeout=5,
                retry_count=3,
                circuit_breaker_enabled=True
            ),
            "database": DependencyConfig(
                url="postgresql://db:5432",
                timeout=30,
                retry_count=1,
                circuit_breaker_enabled=False
            )
        },
        health_checks=[
            HealthCheckConfig(
                name="liveness",
                endpoint="/health/live",
                interval=10,
                timeout=5,
                failure_threshold=3
            )
        ]
    )

    s = str(service)
    assert "user-service" in s
    assert "/users" in s
    assert "auth:" in s
    assert "liveness" in s


def test_data_migration_scenario():
    """Test data model evolution and migration."""
    class UserV1(LLMDataModel):
        id: int
        name: str
        email: str

    class UserV2(LLMDataModel):
        id: int
        first_name: str
        last_name: str
        email: str
        created_at: str | None = None

    v1_data = UserV1(id=1, name="John Doe", email="john@example.com")

    name_parts = v1_data.name.split(" ", 1)
    v2_data = UserV2(
        id=v1_data.id,
        first_name=name_parts[0],
        last_name=name_parts[1] if len(name_parts) > 1 else "",
        email=v1_data.email,
        created_at=datetime.utcnow().isoformat()
    )

    assert v2_data.first_name == "John"
    assert v2_data.last_name == "Doe"
    assert v2_data.email == "john@example.com"


def test_error_response_modeling():
    """Test modeling of error responses."""
    class ErrorDetail(LLMDataModel):
        field: str | None
        message: str
        code: str | None = None

    class ErrorResponse(LLMDataModel):
        error: str
        message: str
        details: list[ErrorDetail] | None = None
        request_id: str | None = None
        timestamp: str

    error = ErrorResponse(
        error="ValidationError",
        message="Invalid input data",
        details=[
            ErrorDetail(field="email", message="Invalid email format", code="INVALID_EMAIL"),
            ErrorDetail(field="age", message="Must be a positive integer", code="INVALID_AGE")
        ],
        request_id="req_abc123",
        timestamp="2024-01-01T00:00:00Z"
    )

    s = str(error)
    assert "ValidationError" in s
    assert "Invalid email format" in s
    assert "INVALID_EMAIL" in s


def test_batch_processing_scenario():
    """Test batch processing with multiple models."""
    class BatchRequest(LLMDataModel):
        batch_id: str
        items: list['ProcessItem']
        options: dict[str, Any]

    class ProcessItem(LLMDataModel):
        id: str
        data: dict[str, Any]
        priority: int = 0

    class BatchResult(LLMDataModel):
        batch_id: str
        processed: int
        failed: int
        results: list['ItemResult']
        duration_ms: float

    class ItemResult(LLMDataModel):
        item_id: str
        success: bool
        result: Any | None = None
        error: str | None = None

    batch_request = BatchRequest(
        batch_id="batch_001",
        items=[
            ProcessItem(id=f"item_{i}", data={"value": i}, priority=i % 3)
            for i in range(5)
        ],
        options={"parallel": True, "timeout": 60}
    )

    batch_result = BatchResult(
        batch_id="batch_001",
        processed=5,
        failed=0,
        results=[
            ItemResult(item_id=f"item_{i}", success=True, result={"processed": i})
            for i in range(5)
        ],
        duration_ms=1234.56
    )

    request_str = str(batch_request)
    assert "batch_001" in request_str
    assert "item_0" in request_str

    result_str = str(batch_result)
    assert "processed: 5" in result_str
    assert "duration_ms: 1234.56" in result_str


def test_streaming_data_model():
    """Test modeling for streaming data scenarios."""
    class StreamMessage(LLMDataModel):
        message_id: str
        stream_id: str
        partition: int
        offset: int
        timestamp: str
        key: str | None
        value: dict[str, Any]
        headers: dict[str, str] | None = None

    class StreamBatch(LLMDataModel):
        stream_id: str
        messages: list[StreamMessage]
        start_offset: int
        end_offset: int
        partition: int

    messages = [
        StreamMessage(
            message_id=f"msg_{i}",
            stream_id="stream_001",
            partition=0,
            offset=i,
            timestamp=f"2024-01-01T00:00:{i:02d}Z",
            key=f"key_{i % 3}" if i % 2 == 0 else None,
            value={"event": f"event_{i}", "data": i},
            headers={"source": "test"} if i % 3 == 0 else None
        )
        for i in range(10)
    ]

    batch = StreamBatch(
        stream_id="stream_001",
        messages=messages,
        start_offset=0,
        end_offset=9,
        partition=0
    )

    s = str(batch)
    assert "stream_001" in s
    assert "msg_0" in s
    assert "end_offset: 9" in s


def test_multi_tenant_configuration():
    """Test multi-tenant configuration modeling."""
    class TenantConfig(LLMDataModel):
        tenant_id: str
        name: str
        features: dict[str, bool]
        limits: dict[str, int]
        custom_settings: dict[str, Any]

    class MultiTenantSystem(LLMDataModel):
        tenants: dict[str, TenantConfig]
        default_features: dict[str, bool]
        global_limits: dict[str, int]

    system = MultiTenantSystem(
        tenants={
            "tenant_a": TenantConfig(
                tenant_id="tenant_a",
                name="Company A",
                features={"feature1": True, "feature2": False},
                limits={"users": 100, "storage_gb": 50},
                custom_settings={"theme": "blue"}
            ),
            "tenant_b": TenantConfig(
                tenant_id="tenant_b",
                name="Company B",
                features={"feature1": True, "feature2": True},
                limits={"users": 500, "storage_gb": 200},
                custom_settings={"theme": "green", "logo_url": "https://example.com/logo.png"}
            )
        },
        default_features={"feature1": True, "feature2": False, "feature3": False},
        global_limits={"max_tenants": 1000, "max_users_per_tenant": 10000}
    )

    s = str(system)
    assert "Company A" in s
    assert "Company B" in s
    assert "feature1: True" in s
    assert "storage_gb: 50" in s
