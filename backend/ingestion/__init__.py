"""
Provide the policy acquisition and ingestion package for RAVIN.

The ingestion package converts current policy source material into
structured, traceable PolicyChunk objects that can be indexed by the
retrieval layer.

Policy identity, headings, source metadata, and text provenance are
preserved so retrieved evidence can later be mapped back to its policy
source.
"""
