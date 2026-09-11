import Link from "next/link";

export default function HomePage() {
  return (
    <main className="container page-content" id="main-content">
      <span className="eyebrow">University policy support</span>
      <h1>Welcome to RAVIN</h1>
      <p className="lead">
        RAVIN searches La Trobe University policy information to provide clear,
        grounded answers with supporting source citations.
      </p>
        <h2>How RAVIN works:</h2>
      <section aria-labelledby="find-answer-title">
        <h3 id="find-answer-title">Find the policy answer you need</h3>
        <p>
          Ask your question in everyday language. RAVIN is designed to identify
          relevant policy material, explain it clearly, and show where the
          information came from.
        </p>
      </section>

      <section aria-labelledby="responsible-title">
        <h3 id="responsible-title">Grounded and transparent</h3>
        <p>
          If sufficient policy evidence is unavailable, RAVIN will tell you
          instead of presenting an unsupported answer.
        </p>
      </section>

      <Link className="primary-link home-cta" href="/question">
        Ask a question <span aria-hidden="true">&rarr;</span>
      </Link>
    </main>
  );
}
