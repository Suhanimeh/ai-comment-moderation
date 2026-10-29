import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";
import * as api from "./api";

jest.mock("./api");

const RESULT = {
  comment: "Great video",
  status: "ok",
  toxicity: "Non-Toxic",
  spam: "Not Spam",
  sentiment: "Positive",
  scores: {
    toxicity: { label: "Non-Toxic", probability: 0.9, uncertain: false },
    spam: { label: "Not Spam", probability: 0.97, uncertain: false },
    sentiment: { label: "Positive", probability: 0.6, uncertain: true },
  },
  all_scores: {},
  model_version: "2.0.0",
};

test("shows per-task probabilities, an uncertainty warning and the model caveat", async () => {
  api.moderateComment.mockResolvedValue(RESULT);
  render(<App />);
  fireEvent.change(screen.getByPlaceholderText(/paste or type/i), { target: { value: "Great video" } });
  fireEvent.click(screen.getByText(/analyze comment/i));
  await waitFor(() => expect(screen.getByText(/90%/)).toBeInTheDocument());
  expect(screen.getByText(/human review advised/i)).toBeInTheDocument();
  expect(screen.getByText(/Model version 2.0.0/i)).toBeInTheDocument();
  expect(screen.queryByText(/^Confidence$/)).not.toBeInTheDocument();
});

test("shows the server message when a comment has no analysable text", async () => {
  api.moderateComment.mockRejectedValue(new Error("Comment contains no analysable text"));
  render(<App />);
  fireEvent.change(screen.getByPlaceholderText(/paste or type/i), { target: { value: "!!!" } });
  fireEvent.click(screen.getByText(/analyze comment/i));
  await waitFor(() => expect(screen.getByText(/no analysable text/i)).toBeInTheDocument());
});
