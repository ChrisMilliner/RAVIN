"use client";

import Link from "next/link";
import { useEffect, useState, useSyncExternalStore } from "react";

const subscribeToQuestion = () => () => {};

function getSavedQuestion() {
  return window.localStorage.getItem("ravinQuestion") || "Your question";
}

export default function ResultView() {
  const question = useSyncExternalStore(
    subscribeToQuestion,
    getSavedQuestion,
    () => "Your question",
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => setIsLoading(false), 1500);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <>
      <section className="question-box" aria-labelledby="submitted-question-title">
        <h2 className="question-label" id="submitted-question-title">
          Question
        </h2>
        <p className="submitted-question">{question}</p>
      </section>

      <section
        className="answer-box"
        aria-live="polite"
        aria-busy={isLoading}
        aria-label="Policy response"
      >
        {isLoading ? (
          <div className="loader" role="status">
            <span className="spinner" aria-hidden="true" />
            <span>Searching policy library...</span>
          </div>
        ) : (
          <>
            <p className="answer-content">
              This is a placeholder answer while the RAVIN policy retrieval
              system is being connected. <b style={{ fontSize: '2rem' }}>🕺</b>
            </p>
            
            <div className="source">
              <strong>Source:</strong> La Trobe University policy library
              placeholder citation.
            </div>
          </>
        )}
      </section>

      <div className="result-actions">
        <Link className="button-link" href="/question">
          Ask another question
        </Link>
      </div>
    </>
  );
}
