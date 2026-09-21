"use client";

import Link from "next/link";
import { useMemo, useSyncExternalStore } from "react";

const subscribeToQuestion = () => () => {};

function getSavedQuestion() {
  return window.localStorage.getItem("ravinQuestion") || "Your question";
}

function getSavedResult() {
  return window.localStorage.getItem("ravinResult") || "";
}

export default function ResultView() {
  const question = useSyncExternalStore(
    subscribeToQuestion,
    getSavedQuestion,
    () => "Your question",
  );
  const savedResult = useSyncExternalStore(
    subscribeToQuestion,
    getSavedResult,
    () => "",
  );

  const result = useMemo(() => {
    if (!savedResult) {
      return null;
    }

    try {
      return JSON.parse(savedResult);
    } catch {
      return null;
    }
  }, [savedResult]);

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
        aria-labelledby="policy-response-title"
      >
        <p className="question-label">Policy response</p>

        <h2 id="policy-response-title">
          {result ? "RAVIN response" : "No response available"}
        </h2>

        {result ? (
          <>
            <p className="answer-content">{result.answer}</p>

            {result.sources?.length ? (
              <div className="source">
                <strong>Sources</strong>

                {result.sources.map((source) => (
                  <p key={`${source.policy_id}-${source.heading}`}>
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {source.title}
                      {source.heading ? ` — ${source.heading}` : ""}
                    </a>
                  </p>
                ))}
              </div>
            ) : null}
          </>
        ) : (
          <p className="answer-content">
            Ask a policy question to generate a RAVIN response.
          </p>
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