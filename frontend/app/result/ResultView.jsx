"use client";

import Link from "next/link";
import { useSyncExternalStore } from "react";

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
        aria-labelledby="prototype-status-title"
      >
        <p className="question-label">Prototype status</p>
        <h2 id="prototype-status-title">Policy-response integration pending</h2>
        <p className="answer-content">
          This page currently demonstrates the question-and-result interface
          only. Policy retrieval and response generation are not connected in
          this prototype.
        </p>
        <p className="field-guidance">
          The UI, API and backend integration is tracked separately under
          COPF-235. No policy answer or citation has been generated.
        </p>
      </section>

      <div className="result-actions">
        <Link className="button-link" href="/question">
          Ask another question
        </Link>
      </div>
    </>
  );
}