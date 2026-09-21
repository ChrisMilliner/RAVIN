import ResultView from "./ResultView";

export const metadata = {
  title: "Policy response",
  description:
    "Review a grounded RAVIN response to a university policy question.",
};

export default function ResultPage() {
  return (
    <main className="result-page" id="main-content">
      <span className="eyebrow">RAVIN response</span>
      <h1>Policy response</h1>
      <ResultView />
    </main>
  );
}
