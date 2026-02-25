# Wikipedia Sample Data Pipeline
#### by Eli Shemuel, PhD

This repository contains a data pipeline for extracting, processing, and augmenting Wikipedia links for use in tasks such as evaluating LLM agents under "controlled confusion" conditions. It leverages Wikipedia2Vec to find semantically similar Wikipedia pages and enriches the dataset with them.

---


## What the Program Does

1. **Loads a pre-labeled dataset** (`df_sampled`) with Wikipedia links in each row.
2. **Parses and merges** all linked Wikipedia URLs from the dataset.
3. **Fetches 2 semantically similar Wikipedia pages** for each original link using [Wikipedia2Vec](https://wikipedia2vec.github.io/).
4. **Filters out rows** where any Wikipedia entity couldn't be resolved in the Wikipedia2Vec embedding space.
5. **Creates a unified list** of:
   - Original Wikipedia links (`merged_wiki_links`)
   - Similar Wikipedia links (`merged_similar_wiki_links`)
   - Additional unrelated links from the dataset (`unused_wiki_links`)
6. **Prepares `total_wiki_links`** as the merged, deduplicated set of all links.
7. Fetches and stores the content of these Wikipedia pages locally.
8. **Saves**:
   7. **Saves**:
   - The enriched DataFrame with prompts, answers, original wiki links, and similar links → `output_data/wiki_sample_prompts.csv`
   - A reproducible list of 90 randomly sampled unrelated wiki links → `output_data/random_sample_links.csv`
---

## 📦 Prerequisites

### 1. Python Dependencies

All required Python packages and versions are listed in:

```
wiki_data_pipeline_reqs.txt
```

To install them, run:

```bash
pip install -r wiki_data_pipeline_reqs.txt
```

This includes:

- `wikipedia2vec`
- `lmdb`
- `mwparserfromhell`

---

### 2. Pretrained Wikipedia2Vec Model

Download the following model file:

📦 [enwiki_20180420_100d.pkl.bz2](https://wikipedia2vec.s3.amazonaws.com/models/enwiki_20180420_100d.pkl.bz2)

Then extract it using:

```bash
bzip2 -d enwiki_20180420_100d.pkl.bz2
```

Place the resulting file (`enwiki_20180420_100d.pkl`) in a the parent directory `Wiki_Sample_Pipeline` known location and update your script with the correct path:
```python
model = Wikipedia2Vec.load("PATH/TO/enwiki_20180420_100d.pkl")
```

---

## 📁 Folder Structure

```
Wiki_Sample_Pipeline/
│
├── Wiki Pages/                     # Where fetched Wikipedia articles are stored
├── wiki_sample_pipeline.py         # The main script
├── wiki_sample_pipeline_reqs.txt   # Dependency list
├── enwiki_20180420_100d.pkl # Pretrained Wikipedia2Vec model file
├── README.md                       # This file
└── output_data/                    # Contains saved outputs: enriched DataFrame and sampled wiki links
```

---

## 📌 Notes & Recommendations

- Use a fixed `random.seed(...)` when sampling to ensure reproducibility.
- Ensure that `wiki_links` in your dataset are valid and follow the pattern `https://en.wikipedia.org/wiki/Title`.
- Handle redirects and special characters (e.g. `AC/DC`, `Titanic_(1997_film)`) with care using Wikipedia API and filename sanitization.
- When writing fetched pages to disk, always sanitize filenames to avoid invalid characters.

---