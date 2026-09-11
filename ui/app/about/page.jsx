import Link from "next/link";

export const metadata = {
  title: "About",
  description:
    "Learn how RAVIN helps La Trobe University students and staff understand policy information.",
};

export default function AboutPage() {
  return (
    <main className="about-page" id="main-content">
      <h1>About RAVIN</h1>
      <p className="lead">
        RAVIN helps La Trobe University students and staff find and understand
        policy information without searching through long documents. Ask a
        question in everyday language and receive a clear answer supported by
        relevant university policy sources.
      </p>

      <section className="feature-grid" aria-label="How RAVIN helps">
        <article className="feature-card">
          <h2>Ask naturally</h2>
          <p>
            Describe your policy question in your own words. You do not need to
            know the policy name or where to find it.
          </p>
        </article>

        <article className="feature-card">
          <h2>Get a clear answer</h2>
          <p>
            RAVIN presents the relevant guidance in straightforward language so
            it is easier to understand and use.
          </p>
        </article>

        <article className="feature-card">
          <h2>Check the source</h2>
          <p>
            Answers include supporting policy references, allowing you to
            confirm the information in the official source.
          </p>
        </article>
      </section>

      <section className="responsible-use" aria-labelledby="responsible-use-title">
        <h2 id="responsible-use-title">Reliable guidance, without guessing</h2>
        <p>
          RAVIN only provides an answer when sufficient policy evidence is
          available. If it cannot find enough supporting information, it will
          say so and direct you towards an appropriate official source or
          university contact. RAVIN provides policy information, not legal
          advice.
        </p>
      </section>

      <div className="about-actions">
        <Link className="primary-link" href="/question">
          Ask a policy question
        </Link>
        <p className="about-note">
          Always review the cited policy before making an important decision.
        </p>
      </div>
    </main>
  );
}
