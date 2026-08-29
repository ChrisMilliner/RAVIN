"""
Provide a development entry point for exercising policy ingestion.

This module supports manual inspection of the acquisition and ingestion
pipeline outside the production answer service. It is intended for
development verification and troubleshooting of source processing.

Production interfaces use the shared RAVIN service bootstrap rather
than this demonstration entry point.
"""

from backend.ingestion.fixtures import INGESTION_FIXTURES
from backend.ingestion.processor import process_policy

def main() -> None:
    """Run the standalone ingestion demonstration over development fixtures.

    The command prints acquisition-independent ingestion outcomes and
    produced chunks for manual inspection. It is a development demonstration
    rather than the production RAVIN application entry point.
    """
    print()
    print("RAVIN Policy Ingestion Reference Pipeline")
    print("=" * 42)

    for policy in INGESTION_FIXTURES:
        print()
        print(f"Policy: {policy.title}")
        print(f"Policy ID: {policy.policy_id}")
        print(f"Status: {policy.status}")
        print(f"Effective date: {policy.effective_date}")
        print(f"Review date: {policy.review_date}")
        print(f"Source: {policy.source_url}")

        result = process_policy(
            policy,
            chunk_size_words=25,
            overlap_words=5,
        )

        print()
        print(f"Ingestion outcome: {result.status.value.upper()}")

        if result.error:
            print(f"Error: {result.error}")

        if result.chunks:
            print(f"Chunks produced: {len(result.chunks)}")

            for chunk in result.chunks:
                print()
                print(f"  Chunk {chunk.chunk_index}")
                print(f"  Status: {chunk.status}")
                print(f"  Effective date: {chunk.effective_date}")
                print(f"  Review date: {chunk.review_date}")
                print(f"  Source: {chunk.source_url}")
                print(f"  Text: {chunk.text}")
        else:
            print("Chunks produced: 0")

        print()
        print("-" * 42)

if __name__ == "__main__":
    main()