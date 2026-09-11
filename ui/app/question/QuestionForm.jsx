"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";

export default function QuestionForm() {
  const router = useRouter();
  const questionRef = useRef(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Enter a policy question before submitting.");
      questionRef.current?.focus();
      return;
    }

    window.localStorage.setItem("ravinQuestion", trimmedQuestion);
    router.push("/result");
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
        <button className="primary-btn" type="submit">
          Submit
        </button>
      </div>
    </form>
  );
}
