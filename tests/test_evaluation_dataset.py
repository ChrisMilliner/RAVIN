import pytest
from backend.evaluation.dataset import (
    load_evaluation_questions,
)

def test_loader_loads_structured_evaluation_questions(
    tmp_path,
):
    dataset_path = tmp_path / "questions.json"

    dataset_path.write_text(
        """
        {
          "questions": [
            {
              "question_id": "Q001",
              "question": "Who approves academic dress changes?",
              "expected_evidence": [
                {
                  "policy_id": "208",
                  "heading_path": [
                    "Section 4 - Key Decisions"
                  ]
                }
              ],
              "notes": "Test question."
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    questions = load_evaluation_questions(
        dataset_path
    )

    assert len(questions) == 1
    assert questions[0].question_id == "Q001"
    assert questions[0].expected_evidence[0].policy_id == "208"

def test_loader_rejects_empty_question_set(
    tmp_path,
):
    dataset_path = tmp_path / "questions.json"

    dataset_path.write_text(
        """
        {
          "questions": []
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation dataset cannot be empty.",
    ):
        load_evaluation_questions(dataset_path)

def test_loader_rejects_invalid_evidence_structure(
    tmp_path,
):
    dataset_path = tmp_path / "questions.json"

    dataset_path.write_text(
        """
        {
          "questions": [
            {
              "question_id": "Q001",
              "question": "Test question",
              "expected_evidence": [
                {
                  "policy_id": "208",
                  "heading_path": "Section 4"
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected evidence heading path must be a list."
        ),
    ):
        load_evaluation_questions(dataset_path)