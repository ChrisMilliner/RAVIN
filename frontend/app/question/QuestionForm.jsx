"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { askRavin } from "../../lib/ravinApi";

export default function QuestionForm() {
  const router = useRouter();
  const questionRef = useRef(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Enter a policy question before submitting.");
      questionRef.current?.focus();
      return;
    }

    setError("");
    setIsSubmitting(true);

    try {
      const result = await askRavin(trimmedQuestion);

      window.localStorage.setItem(
        "ravinQuestion",
        trimmedQuestion,
      );

      window.localStorage.setItem(
        "ravinResult",
        JSON.stringify(result),
      );

      router.push("/result");
    } catch (requestError) {
      setError(
        requestError.message ||
          "RAVIN could not process your question. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="question-box">
        <label className="question-label" htmlFor="question">
          Question
        </label>
        <textarea
          ref={questionRef}
          id="question"
          name="question"
          value={question}
          onChange={(event) => {
            setQuestion(event.target.value);
            if (error) setError("");
          }}
          placeholder="Type your question here..."
          aria-describedby={error ? "question-error" : "question-guidance"}
          aria-invalid={Boolean(error)}
        />
        <p className="field-guidance" id="question-guidance">
          Include useful details, but do not enter passwords or other sensitive
          personal information.
        </p>
        {error ? (
          <p className="form-error" id="question-error" role="alert">
            {error}
          </p>
        ) : null}
      </div>

      <div className="actions">
        <button
          className="secondary-btn"
          type="button"
          onClick={() => router.push("/")}
        >
          Cancel
        </button>
        <button
          className="primary-btn"
          type="submit"
          disabled={isSubmitting}
        >
          {isSubmitting ? "Finding answer..." : "Submit"}
        </button>
      </div>
    </form>
  );
}
