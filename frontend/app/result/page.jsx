import ResultView from "./ResultView";

export const metadata = {
  title: "Prototype Status",
  description:
    "Review a submitted question in the RAVIN interface prototype. Policy-response integration is pending.",
};

export default function ResultPage() {
  return (
    <main className="result-page" id="main-content">
      <span className="eyebrow">UI prototype</span>
      <h1>Policy response</h1>
      <ResultView />
    </main>
  );
}
