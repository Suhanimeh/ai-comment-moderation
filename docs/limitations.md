# Limitations and responsible use

## Data and generalisation

- **Domain mismatch.** The three classifiers are trained on tweets (toxicity, sentiment) and SMS (spam), not on YouTube/Instagram comments. Reported metrics describe performance on *those* held-out samples only. Performance on other platforms is unmeasured and is likely lower.
- **Different data per task.** A single comment is scored by three models trained on three unrelated datasets. The three outputs are independent; no combined score is provided.
- **Small test sets.** Sentiment has 662 test examples (101 Positive, 112 Negative), spam has 128 Spam examples. With one split and seed, small differences between models are not statistically established.
- **Label noise.** Labels come from crowd annotation; some "neutral" sentiment and "offensive" toxicity labels are debatable (see `docs/methodology.md`).
- **Sentiment is weak** (macro-F1 0.618). Treat it as a rough signal, not a decision input.

## Bias and fairness

- The toxicity data is dominated by the Toxic class (about 83%) and comes from a corpus where "offensive language" includes slang. Published work on this kind of data reports that dialect and identity terms can be over-flagged. **I did not measure bias or per-group error rates**, so the absence of a finding is not evidence of fairness.
- Word-level TF-IDF features cannot distinguish quoting, reclaiming, or discussing offensive language from using it, so false positives on such content are expected.

## Error types

- **False positives** (benign comment flagged Toxic/Spam) can silence legitimate speech. Toxicity: 46 of 832 Non-Toxic test comments were flagged (5.5%).
- **False negatives** (harmful comment missed) leave abuse visible. Toxicity: 187 of 4,113 Toxic test comments were missed (4.5%); spam: 6 of 128.
- Obfuscated text (leetspeak, spacing, misspellings), sarcasm, and non-English text are not handled specifically; mixed-language input is preserved by preprocessing but the models were trained on English only.

## Probabilities

- Probabilities are the model's estimate for its predicted class, not guarantees. Logistic Regression and the SVC were trained with balanced class weights, so probabilities do not reflect real-world class frequencies, and they were not separately calibrated on held-out data.
- The "uncertain" flag is task-specific; for the binary tasks it never triggers (see methodology).

## Privacy

- Moderation requests are logged with timestamp, length, labels, per-task probabilities and model version. **Raw comment text is not stored by default** (`STORE_RAW_COMMENT=false`). If you enable it, define a retention policy first.
- Comment text is not written to application logs; errors log only the input length.

## Security status (proof-of-concept)

Implemented: input validation, explicit CORS origins, generic error messages, per-IP rate limiting (process-local), readiness checks. **Not implemented:** authentication/authorisation, shared-store rate limiting, audit logging, dependency pinning/scanning, load testing, penetration testing. Do not describe this service as production-secure.

## Human oversight

Automated moderation makes mistakes. Use predictions to prioritise human review, not as the sole basis for removing content or penalising people.

## Model file safety

`moderation_model.joblib` is a pickle-based artifact. Loading an untrusted joblib file can execute arbitrary code. Only load files you trained yourself.
