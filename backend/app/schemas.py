from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class LoginInput(ApiModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128, repr=False)


class PublicUser(ApiModel):
    id: UUID
    username: str
    display_name: str


class ErrorDetails(ApiModel):
    code: str
    message: str
    details: dict | None
    request_id: str


class ErrorResponse(ApiModel):
    error: ErrorDetails


class Files(ApiModel):
    original: str
    mask: str | None
    overlay: str | None


class Analysis(ApiModel):
    id: UUID
    filename: str
    model_id: str
    model_name: str
    model_version: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    processing_ms: int | None
    width: int
    height: int
    error_message: str | None
    error_code: str | None
    queue_position: int | None
    processing_stage: Literal["preprocessing", "inference", "postprocessing"] | None
    files: Files


class AnalysisPage(ApiModel):
    items: list[Analysis]
    page: int
    page_size: int
    total: int


class SegmentationModel(ApiModel):
    id: str
    name: str
    version: str
    description: str
    is_default: bool
    available: bool
