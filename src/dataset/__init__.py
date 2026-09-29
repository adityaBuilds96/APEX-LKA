"""
src/dataset/__init__.py
=======================
Dataset ingestion, validation, reporting, and annotation management.
"""

from src.dataset.report import DatasetReport
from src.dataset.validator import DatasetValidator
from src.dataset.ingestion import DatasetIngestor
from src.dataset.annotation_prep import AnnotationWorkspace, CLASS_DEFINITIONS
from src.dataset.auto_annotator import ClassicalAutoAnnotator, AutoAnnotationMetadata, AutoAnnotationResult
from src.dataset.quality_scorer import QualityScorer, QualityScoreReport, CheckResult, CheckSeverity, QualityTier
from src.dataset.thumbnail_generator import ThumbnailGenerator
from src.dataset.batch_processor import BatchProcessor, BatchItem, BatchProgress, ItemStatus
from src.dataset.pipeline_orchestrator import PipelineOrchestrator, PipelineState, PipelineExecutionSummary

__all__ = [
    "DatasetReport",
    "DatasetValidator",
    "DatasetIngestor",
    "AnnotationWorkspace",
    "CLASS_DEFINITIONS",
    "ClassicalAutoAnnotator",
    "AutoAnnotationMetadata",
    "AutoAnnotationResult",
    "QualityScorer",
    "QualityScoreReport",
    "CheckResult",
    "CheckSeverity",
    "QualityTier",
    "ThumbnailGenerator",
    "BatchProcessor",
    "BatchItem",
    "BatchProgress",
    "ItemStatus",
    "PipelineOrchestrator",
    "PipelineState",
    "PipelineExecutionSummary",
]
