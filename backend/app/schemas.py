from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Files(ApiModel):
    original: str
    mask: str | None
    overlay: str | None


class Analysis(ApiModel):
    id: UUID
    filename: str
    model_id: str
    model_name: str = "Test infrastruktury — bez modelu AI"
    model_version: str = "0.1.0"
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
