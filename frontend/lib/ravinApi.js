export async function askRavin(question) {
  const ravinApiUrl =
    process.env.NEXT_PUBLIC_RAVIN_API_URL ||
    "http://127.0.0.1:8000";
  const response = await fetch(`${ravinApiUrl}/api/questions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
    }),
  });

  let data = {};

  try {
    data = await response.json();
  } catch {
    // Use the generic error below if the API response is not JSON.
  }

  if (!response.ok) {
    throw new Error(
      data.message ||
        "RAVIN could not process your question. Please try again.",
    );
  }

  return data;
}