# Data

Raw datasets are **not committed**. Download them with:

```bash
python scripts/download_data.py      # writes data/raw/
```

| Task | File in `data/raw/` | Source | Size used | Label mapping |
|---|---|---|---|---|
| Toxicity | `davidson_labeled_data.csv` | Davidson et al. (2017), *Automated Hate Speech Detection and the Problem of Offensive Language* - [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language) (tweets) | 24,723 after de-dup | `hate_speech` or `offensive_language` -> **Toxic**; `neither` -> **Non-Toxic** |
| Spam | `sms_spam.tsv` | SMS Spam Collection (Almeida & Hidalgo, UCI ML Repository), mirrored in [justmarkham/pycon-2016-tutorial](https://github.com/justmarkham/pycon-2016-tutorial) | 5,157 after de-dup | `spam` -> **Spam**; `ham` -> **Not Spam** |
| Sentiment | `sanders_full_corpus.csv` | Sanders Analytics Twitter sentiment corpus, mirrored in [zfz/twitter_corpus](https://github.com/zfz/twitter_corpus) | 3,306 after de-dup | `positive` / `negative` / `neutral` kept; `irrelevant` dropped |

## Preparation (see `backend/data_loading.py`)

1. HTML entities are decoded; empty/NaN rows dropped.
2. Exact duplicates are removed (after lowercasing, stripping a leading `RT @user:` and collapsing whitespace) **before** splitting, so identical texts cannot land in both train and test.
3. Stratified 80/20 train/test split, seed 42.

## Licences and provenance - please verify

I did not independently verify the redistribution licences of these datasets, which is another reason the raw files are not committed. Check each source repository / UCI page before redistributing, and cite the original papers if you publish results. Dataset SHA-256 hashes are recorded in the model metadata (`experiments/model_metadata.json`).

## Known data limitations

- The three tasks use **three different datasets from three different domains** (tweets and SMS), not YouTube/Instagram comments. See `docs/limitations.md`.
- Toxicity labels come from crowd annotation of *tweets*; "offensive" includes slang and reclaimed language that is not necessarily abusive, and the Toxic class is the majority (~83%).
- The sentiment set is small (3,306) and its "neutral" class includes many news-like or product-announcement tweets.
