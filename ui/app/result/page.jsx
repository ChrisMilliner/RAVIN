import ResultView from "./ResultView";

export const metadata = {
  title: "Result",
  description: "Review a prototype RAVIN policy response and its source citation.",
};

export default function ResultPage() {
  return (
    <main className="result-page" id="main-content">
      <span className="eyebrow">Answer</span>
      <h1>Response</h1>
      <ResultView />
    </main>
  );
}
