from backend.core.fixtures import POLICY_FIXTURES
from backend.core.response import build_grounded_response

def main() -> None:
    print()
    print("RAVIN Grounded Response Reference Slice")
    print("=" * 40)
    print()
    print(
        "This demonstration uses synthetic policy fixtures "
        "and deterministic retrieval."
    )
    print()

    question = input("Question: ")

    try:
        response = build_grounded_response(
            question,
            POLICY_FIXTURES,
        )
    except ValueError as error:
        print()
        print(f"Input error: {error}")
        return

    print()
    print(f"Outcome: {response.outcome.value.upper()}")
    print()

    if response.answer:
        print("Response:")
        print(response.answer)

    if response.sources:
        print()
        print("Supporting sources:")

        for source in response.sources:
            print()
            print(f"- {source.policy_title}")
            print(f"  Policy ID: {source.policy_id}")
            print(f"  Source: {source.source_url}")
            print(f"  Retrieval score: {source.relevance_score:.2f}")
    else:
        print()
        print("Supporting sources: None")

    print()

if __name__ == "__main__":
    main()