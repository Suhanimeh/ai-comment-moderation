# Methodology and evaluation report

All numbers below are **measured** by `backend/train.py` / `backend/evaluate.py` and reproduced in
[`experiments/evaluation_summary.md`](../experiments/evaluation_summary.md). Re-run the scripts to regenerate them.

## 1. Research question

Which classical text-classification approach (Logistic Regression, Multinomial Naive Bayes, or a linear SVM) performs best on each of three moderation tasks - toxicity, spam and sentiment - when features are TF-IDF word uni/bi-grams, and does keeping URL/number indicators during preprocessing help?

## 2. Method

| Step | Choice |
|---|---|
| Data | Three public datasets, one per task (`data/README.md`), de-duplicated before splitting |
| Split | Stratified 80/20 train/test, seed 42 |
| Features | TF-IDF, word 1-2-grams, `min_df=2`, 30,000 features, sublinear tf. The vocabulary is learned inside each `Pipeline`, so only training folds are used (no leakage) |
| Preprocessing | `legacy` (delete URLs/mentions/digits) vs `enhanced` (indicator tokens, collapse repeated characters); chosen per task by 5-fold CV on the training split |
| Models | Logistic Regression (baseline), Multinomial Naive Bayes, Linear SVC wrapped in `CalibratedClassifierCV` so it yields probabilities |
| Tuning | `GridSearchCV`, 5-fold stratified, scoring **macro-F1**, on the training split only |
| Class imbalance | `class_weight="balanced"` for LR and SVC (Naive Bayes has no such option) |
| Selection | Highest CV macro-F1. The test set did **not** influence selection; it is scored once per candidate for reporting |
| Deployed model | The selected pipeline fitted on the training split (so deployed behaviour matches the reported test metrics) |
| Uncertainty flag | Threshold chosen from out-of-fold *training* predictions, then checked on the test set |

## 3. Results (held-out test set)

| Task | Selected model | Accuracy | Macro-F1 | Key class | Key-class precision / recall |
|---|---|---:|---:|---|---|
| Toxicity (4,945 test texts) | Logistic Regression | 0.953 | 0.921 | Toxic | 0.988 / 0.955 |
| Spam (1,032) | Logistic Regression | 0.991 | 0.980 | Spam | 0.976 / 0.953 |
| Sentiment (662) | Logistic Regression | 0.709 | 0.618 | Negative | 0.543 / 0.562 |

What the comparison shows:

- **Logistic Regression had the highest CV macro-F1 on all three tasks**, but on toxicity and spam the three models are close; the differences between LR and the calibrated SVC are small (e.g. toxicity test macro-F1 0.921 vs 0.919) and with one split and one seed I would not claim a real winner there. The clear result is that Multinomial Naive Bayes was worse on toxicity (macro-F1 0.823).
- **Accuracy hides the weak spots.** On toxicity the *Non-Toxic* minority class has precision 0.808 / recall 0.945: about 1 in 5 comments the model passes as non-toxic is actually toxic, a weakness the strong Toxic-class numbers (precision 0.988) do not show. On sentiment, accuracy (0.709) is flattered by the large Neutral class; macro-F1 is 0.618 and the Positive/Negative classes score far lower.
- **Naive Bayes had the *highest* sentiment accuracy (0.751) but lower macro-F1 (0.600)** because it favours the Neutral majority - a direct illustration of why accuracy alone should not be used to rank these models.

### Preprocessing ablation (5-fold CV, training split)

| Task | legacy | enhanced |
|---|---:|---:|
| Toxicity | 0.9082 ± 0.0062 | 0.9084 ± 0.0047 |
| Spam | 0.9363 ± 0.0106 | **0.9506 ± 0.0034** |
| Sentiment | **0.5918 ± 0.0104** | 0.5886 ± 0.0166 |

Keeping URL/number indicator tokens helped **spam** (+0.014 macro-F1, larger than the fold std), which supports the hypothesis that deleting URLs and digits discards spam signal. For toxicity and sentiment the differences are within noise; the code picks the higher mean, but this is not evidence that either variant is better there.

### Uncertainty threshold

The "low confidence" flag uses the smallest top-class probability `t` (grid 0.50-0.95) where out-of-fold training predictions at or above `t` were >= 90% accurate while covering >= 30% of data.

- Toxicity and spam: `t = 0.50`, i.e. **the flag can never fire for these binary tasks** (the top class always has p >= 0.5). Overall accuracy was already above the 90% target, so no extra abstention was warranted by this rule. This does not mean individual predictions are reliable - see error analysis.
- Sentiment: the 90% target is not reachable, so the fallback gives `t = 0.75`. On the test set, 37.8% of predictions are at/above 0.75 with accuracy 0.872; the remaining 62.2% (flagged) have accuracy 0.609. The flag is therefore informative for sentiment.

## 4. Error analysis

Taken from the most confident misclassifications on the test set (`experiments/error_analysis_<task>.csv`). Explanations are my hypotheses from reading the examples; I did not run experiments to confirm them.

| Task | Example type (paraphrased from real test errors) | Actual -> Predicted | Probable explanation |
|---|---|---|---|
| Spam | Promotional text with no typical spam keywords (a "new downloads for our members, click the link" message) | Spam -> Not Spam | Obfuscated / unusual wording; spam vocabulary evolves |
| Spam | A genuine message that mentions a bill amount and a phone number | Not Spam -> Spam | Numeric and currency tokens are strong spam indicators in the training data, and the `enhanced` preprocessing preserves exactly those indicators |
| Spam | A spam message that reads like a casual personal text ("sorry I missed your call... I'm on <number>") | Spam -> Not Spam | Conversational style resembles ham; only the phone number hints at spam |
| Sentiment | Tweets thanking a brand but labelled Neutral (e.g. a joke about a voice assistant) | Neutral -> Positive | Sentiment words ("thanks", "likes") dominate; annotator judged the tweet informational/humorous. Label ambiguity, not necessarily a model failure |
| Sentiment | A tweet reporting a website error | Negative -> Neutral | No explicit negative vocabulary; negativity is implied by the situation |
| Sentiment | News-style link tweets | Positive -> Neutral | URL + hashtag text carries almost no polarity signal |
| Toxicity | Casual insults with no profanity applied to sports players / things ("X so trash") | Toxic -> Non-Toxic | The label comes from annotator judgement of *offensive* tweets; the vocabulary overlaps heavily with benign text |
| Toxicity | Short tweets containing a slang word common in the Toxic class but labelled neither | Non-Toxic -> Toxic | Word-level features cannot separate reclaimed/casual use from abuse |

**Domain-shift check (not a measured result).** Because these classifiers are trained on tweets/SMS, I also tried a handful of hand-written YouTube-style comments. One of them - "I do not like this at all" - was classified **Toxic with probability 0.50**, which is a clear false positive, because the toxicity training data mostly contains profanity-heavy tweets and little polite negative criticism. This is an anecdote, not a benchmark, but it matches the expected weakness (see `limitations.md`).

## 5. Limitations of these conclusions

See [`limitations.md`](limitations.md). In short: one train/test split and one random seed (no confidence interval on test metrics), small test sets for spam and sentiment, label noise, and training/deployment domain mismatch.

## 6. Reproducing

```bash
pip install -r backend/requirements.txt
python scripts/setup_nltk.py
python scripts/download_data.py
cd backend && python train.py        # ~2 minutes on one CPU core
python evaluate.py                    # re-evaluates the saved model on the same split
```
