// api.js
// Fetch API integration for the AI Comment Moderation backend.
// Set REACT_APP_API_URL in your .env file for production.

const BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

/**
 * Analyze a comment through the moderation API.
 * @param {string} comment
 * @returns {Promise<{comment, status, toxicity, spam, sentiment, scores, all_scores, model_version}>}
 * `scores[task]` = {label, probability, uncertain}. HTTP 422 (empty or no analysable
 * text) surfaces as an Error carrying the server's `detail` message.
 */
export async function moderateComment(comment) {
  const response = await fetch(`${BASE_URL}/moderate-comment`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ comment }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const detail = Array.isArray(err.detail) ? err.detail.map((d) => d.msg).join("; ") : err.detail;
    throw new Error(detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Health-check the backend.
 * @returns {Promise<{status: string}>}
 */
export async function healthCheck() {
  const response = await fetch(`${BASE_URL}/ready`);
  return response.json();
}
