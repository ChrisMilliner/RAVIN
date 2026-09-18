import QuestionForm from "./QuestionForm";

export const metadata = {
  title: "Ask a question",
  description: "Ask RAVIN a natural-language question about university policy.",
};

export default function QuestionPage() {
  return (
    <main className="question-page" id="main-content">
      <span className="eyebrow">Prompt</span>
      <h1>Ask a policy question</h1>
      <p>
        Share the exact question you want answered, including any relevant
        context that may help RAVIN locate the right policy information.
      </p>
      <QuestionForm />
    </main>
  );
}
