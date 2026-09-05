"""
Define configuration used by the RAVIN policy-ingestion pipeline.

This module centralises ingestion settings so acquisition and processing
behaviour can be adjusted without embedding configuration decisions
throughout the ingestion implementation.

The configuration belongs to source preparation and does not determine
retrieval ranking or answer behaviour.
"""

DEFAULT_CHUNK_SIZE_WORDS = 100
DEFAULT_CHUNK_OVERLAP_WORDS = 20