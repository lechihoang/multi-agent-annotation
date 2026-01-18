"""
Base data models for the Multi-Agent Annotation System.

This module defines all shared data types used across the system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ===========================================
# Enums
# ===========================================


class ComplexityLevel(str, Enum):
    """Task complexity classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DecisionType(str, Enum):
    """Routing decision after consensus."""

    APPROVE = "approve"
    REVIEW = "review"
    ESCALATE = "escalate"


class TaskStatus(str, Enum):
    """Task processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    IN_REVIEW = "in_review"


class ReviewStatus(str, Enum):
    """Human review status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


# ===========================================
# Core Models
# ===========================================


class Task(BaseModel):
    """A task to be annotated."""

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., description="The text content to annotate")
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    complexity: Optional[ComplexityLevel] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Entity(BaseModel):
    """An extracted entity."""

    text: str = Field(..., description="The entity text")
    type: str = Field(..., description="Entity type (PERSON, ORG, LOC, etc.)")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    confidence: float = Field(..., ge=0.0, le=1.0)


class IntentAnnotation(BaseModel):
    """Intent classification result."""

    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    alternatives: list[dict[str, float]] = Field(default_factory=list)


class EntityAnnotation(BaseModel):
    """Entity extraction result."""

    entities: list[Entity] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class FAQAnnotation(BaseModel):
    """FAQ matching result."""

    matched_faq: Optional[str] = None
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: Optional[str] = None


# ===========================================
# Agent Annotation Models
# ===========================================


class Annotation(BaseModel):
    """Base annotation from any agent."""

    agent_name: str = Field(..., description="Name of the agent")
    confidence: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    result: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Vote(BaseModel):
    """A vote from an agent in the consensus process."""

    agent_name: str
    confidence: float
    weight: float
    weighted_score: float


# ===========================================
# Consensus Models
# ===========================================


class ConsensusResult(BaseModel):
    """Result of the consensus voting process."""

    score: float = Field(..., ge=0.0, le=1.0, description="Weighted consensus score")
    decision: DecisionType = Field(..., description="Routing decision")
    votes: list[Vote] = Field(default_factory=list)
    audit_trail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# Annotation Result
# ===========================================


class AnnotationResult(BaseModel):
    """Complete annotation result for a task."""

    task_id: UUID
    intent: Optional[IntentAnnotation] = None
    entities: Optional[EntityAnnotation] = None
    faq: Optional[FAQAnnotation] = None
    consensus: Optional[ConsensusResult] = None
    annotations: list[Annotation] = Field(default_factory=list)
    status: TaskStatus = Field(default=TaskStatus.COMPLETED)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# Human Review Models
# ===========================================


class ReviewItem(BaseModel):
    """An item in the human review queue."""

    id: UUID = Field(default_factory=uuid4)
    task: Task
    annotation_result: AnnotationResult
    consensus_score: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(default=0)
    status: ReviewStatus = Field(default=ReviewStatus.PENDING)
    assigned_to: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None


class HumanReview(BaseModel):
    """Human reviewer's decision."""

    review_item_id: UUID
    reviewer_id: str
    decision: ReviewStatus
    corrections: dict[str, Any] = Field(default_factory=dict)
    comments: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# API Response Models
# ===========================================


class ChatResponse(BaseModel):
    """Response from an LLM API call."""

    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0)


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    providers: dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# Metrics Models
# ===========================================


class MetricsSnapshot(BaseModel):
    """Snapshot of system metrics."""

    annotation_count: int = 0
    automation_rate: float = 0.0
    average_consensus_score: float = 0.0
    review_queue_size: int = 0
    api_latency_p50_ms: float = 0.0
    api_latency_p95_ms: float = 0.0
    error_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
