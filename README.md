# AirPanel

AirPanel is a research workspace for analysing synthetic consumer interviews about two telecom advertisements: Bouygues Telecom and Orange. The project combines LLM-coded annotations, score extraction, lexical analysis, embeddings, demographic faceting, model comparison, and sample-size stability checks.

This README is a quick file guide for reviewing what each notebook or script does.

## Project Logic

```text
interviews.csv
  -> clean / inspect interviews
  -> annotate preferences and scores with local LLMs
  -> merge annotations back into panel-level datasets
  -> analyse results by brand, dimension, demographics, text, embeddings, and sampling stability
```

Main research dimensions:

```text
creativity
humour
originality
reliability_trust
intent_to_purchase
overall / overallpreference
informativeness
expressivity
```

Main brands:

```text
Bouygues
Orange
```

## Required Data

The Git repository contains notebooks, reports, and scripts, but most large CSV files are expected to be placed in the `AirPanel/` folder.

| File | Role |
|---|---|
| `interviews.csv` | Main synthetic interview panel. One row per panelist. Contains demographics and all interview answers. |
| `human_interviews.csv` | Raw human-interview comparison file. |
| `humans_interviews.csv` | Cleaned human-interview file generated from `human_interviews.ipynb`. |
| `df_allq.csv` | Long-format question-answer table. One row per panelist-question answer. |
| `embedding_allq.csv` | Embeddings corresponding to `df_allq.csv`. |
| `preference_annotations_extended.csv` | Categorical LLM annotations by dimension. |
| `interviews_with_extended_preferences.csv` | `interviews.csv` merged with categorical annotations. |
| `preference_annotations_scores.csv` | Raw score annotations stored as dictionaries per dimension. |
| `interviews_with_scores.csv` | Flattened score annotations merged with interviews. Best file for score modelling. |
| `preference_annotations_informativeness_expressivity.csv` | Raw informativeness/expressivity score annotations. |
| `interviews_with_informativeness_expressivity_scores.csv` | Flattened informativeness/expressivity scores merged with interviews. |

## File Map

### Core Exploration

| File | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `airpanel_exploratory.ipynb` | Initial inspection of the synthetic interview data. Checks columns, question structure, and answer format. | `interviews.csv`, `df_allq.csv` | Exploratory tables and checks. |
| `human_interviews.ipynb` | Cleans and explores human-interview data for comparison with synthetic interviews. | `human_interviews.csv` | `humans_interviews.csv`, human demographic/preference plots, `human_interviews_nlp.csv`. |

### LLM Annotation

| File | What it does | Model | Main outputs |
|---|---|---|---|
| `further_analysis.ipynb` | Early multi-dimensional categorical preference annotation. Covers creative, commercial, and overall preference. | Mistral / Mistral-7B style local annotation | `preference_annotations_multidim.csv`, `interviews_with_multidim_preferences.csv`, early preference plots. |
| `facetting.ipynb` | Main extended categorical annotation and demographic faceting. Labels each panelist by dimension: creativity, humour, originality, reliability/trust, intent to purchase, overall preference. | local `mistral` via Ollama | `preference_annotations_extended.csv`, `interviews_with_extended_preferences.csv`, `preference_summary_extended.csv`, `extended_facet_table.csv`, `charts/`. |
| `score.ipynb` | Main 0-10 score annotation. Scores Bouygues and Orange separately for each dimension. | local `mistral` via Ollama, temperature `0.1` | `preference_annotations_scores.csv`, `interviews_with_scores.csv`, `score_summary.csv`, `score_facet_charts/`. |
| `p16_informativeness_expressivity_detection.ipynb` | Adds 0-10 scores for informativeness and expressivity. | `mistral:7b` via Ollama, temperature `0.1` | `preference_annotations_informativeness_expressivity.csv`, `interviews_with_informativeness_expressivity_scores.csv`, `informativeness_expressivity_summary.csv`, `informativeness_expressivity_facet_charts/`. |
| `models.ipynb` | Generalized annotation runner for several model/approach combinations. Useful for robustness checks. | `mistral:7b`, `gemma3`, `gemma4` | `models_outputs/`, `run_manifest.csv`, per-model summaries and quality reports. |

### Text and Lexical Analysis

| File | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `adjectives.ipynb` | Extracts and compares adjectives associated with each brand. | `interviews.csv`, older preference/adjective files | `panel_adjectives_extracted.csv`, adjective wordclouds, adjective bar charts. |
| `words_analysis_interviews.ipynb` | Larger POS-based lexical analysis: adjectives, nouns, verbs, richness, overlaps, brand vocabularies. | `interviews.csv`, `panel_adjectives_extracted.csv` | wordclouds, token-frequency charts, Jaccard/richness plots. |

### Embeddings

| File | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `embeddings_answers_comparison.ipynb` | Builds semantic diagnostics from answer embeddings. Compares Bouygues, Orange, and comparison-answer spaces. | `df_allq.csv`, `embedding_allq.csv` | `embedding_comparison_outputs/`, including `df_allq_with_embeddings_metadata.csv`. |
| `embeddings_professional_inspection.ipynb` | Inspects embedding quality and structure more systematically. Produces executive diagnostics and plots. | `embedding_comparison_outputs/` | `embedding_inspection_outputs/`, including cohesion, similarity, PCA, and distinctiveness summaries. |

### Stability / Sample Size

| File | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `kl_sampling_progression.py` | Command-line script for estimating how many responses are needed for stable brand-winner conclusions. Uses repeated subsampling and KL divergence. | `preference_annotations_extended.csv` | `kl_sampling_progression_outputs/`: replicates, summary, decision table, plot, run metadata. |
| `README_kl_sampling_progression.md` | Short usage note for `kl_sampling_progression.py`. | none | command examples and output explanation. |
| `kl_sampling_progression_plot_reps2000.png` | Example KL/sample-size stability plot from a high-repetition run. | generated plot | visual reference. |

### Reports

| File | Content |
|---|---|
| `Progress_Report_1.pdf` | Early methodological framing: metrics, embeddings, divergence, and evaluation ideas. |
| `Progress_Report_2.pdf` | Adjective-distance analysis between Orange and Bouygues. |
| `Progress_Report_3_compressed.pdf` | Broader lexical analysis with adjectives, nouns, verbs, wordclouds, and token distributions. |
| `Progress_Report_4.pdf` | Multi-dimensional LLM preference annotation with Mistral-7B. |
| `Progress_Report_5.pdf` | Demographic chart checkpoint for preference dimensions. |
| `Progress_Report_6.pdf` | Main brand-perception progress report. Summarizes brand differences by dimension and segment. |
| `Progress_Report_6_beamer.pdf` | Presentation version of report 6. |
| `short_note_humans_vs_interviews.pdf` | Human/synthetic lexical comparison note using adjective subsamples. |

## Most Important Generated Datasets

### Categorical Preference Labels

```text
preference_annotations_extended.csv
```

One row per panelist. Contains final LLM-coded labels:

```text
creativity
humour
originality
reliability_trust
intent_to_purchase
overallpreference
```

Labels are typically:

```text
Bouygues / Orange / Mixed / Neutral
```

### Continuous Scores

```text
interviews_with_scores.csv
```

Best file for score-based analysis. Contains interview data plus flattened score columns:

```text
creativity_bouygues
creativity_orange
humour_bouygues
humour_orange
originality_bouygues
originality_orange
reliability_trust_bouygues
reliability_trust_orange
intent_to_purchase_bouygues
intent_to_purchase_orange
overall_bouygues
overall_orange
```

### Raw Score Annotations

```text
preference_annotations_scores.csv
```

Same scoring information as above, but stored as dictionaries inside each dimension:

```text
creativity = {"Bouygues": 9, "Orange": 5}
humour = {"Bouygues": 8, "Orange": 4}
```

### Embedding Metadata

```text
embedding_comparison_outputs/df_allq_with_embeddings_metadata.csv
```

Long-format answer-level dataset with:

```text
panelist_id
Question
text
group
family
text_len
emb_0 ... emb_383
```

This is the key file for question-level embedding or text-signal modelling.

## How To Run The Most Reproducible Script

From the `AirPanel/` folder:

```bash
python kl_sampling_progression.py
```

More precise run:

```bash
python kl_sampling_progression.py --reps 2000 --output_dir kl_sampling_progression_outputs_precise
```

Run on selected dimensions:

```bash
python kl_sampling_progression.py --dims overallpreference,intent_to_purchase,reliability_trust
```

The script writes:

```text
kl_sampling_progression_replicates.csv
kl_sampling_progression_summary.csv
kl_sampling_progression_decision_table.csv
kl_sampling_progression_plot.png
kl_sampling_progression_run.json
```

## Environment

Recommended Python packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly kaleido wordcloud requests
pip install spacy sentence-transformers transformers torch
python -m spacy download fr_core_news_sm
python -m spacy download fr_core_news_md
python -m spacy download fr_core_news_lg
```

For local LLM annotation:

```bash
ollama serve
ollama pull mistral:7b
ollama pull gemma3
ollama pull gemma4
```

Some notebooks call the model `mistral`; others call `mistral:7b`. Check the local Ollama tag before rerunning annotations.

## Suggested Reading Order

For quick supervisor review:

```text
1. score.ipynb
2. facetting.ipynb
3. embeddings_answers_comparison.ipynb
4. embeddings_professional_inspection.ipynb
5. kl_sampling_progression.py
6. Progress_Report_6.pdf
```

For full project history:

```text
1. Progress_Report_1.pdf
2. Progress_Report_2.pdf
3. Progress_Report_3_compressed.pdf
4. Progress_Report_4.pdf
5. Progress_Report_5.pdf
6. Progress_Report_6.pdf
```

## Current Methodological Status

What the repository supports:

- LLM-coded preference labels by dimension.
- LLM-coded 0-10 scores by brand and dimension.
- Demographic faceting of labels and scores.
- Lexical comparison between brand responses.
- Embedding-based comparison of answer spaces.
- Sample-size stability analysis using KL divergence and winner stability.
- Exploratory comparison between synthetic and human interview language.

What it does not yet fully contain:

- A clean package-style pipeline.
- A committed copy of all required large CSV data.
- Direct per-question attribution scores such as `creativity_q1_bouygues`.
- Human-validated annotation reliability for all dimensions.
- A fully documented original embedding-generation step.

## Recommended Next Steps

| Priority | Task |
|---|---|
| High | Add a `data/README.md` listing every required input file and every generated output file. |
| High | Convert main annotation notebooks into scripts with fixed configs. |
| High | Save model name, prompt version, temperature, and input hash for every annotation run. |
| Medium | Add per-question-per-dimension annotation if the goal is direct question contribution analysis. |
| Medium | Add `requirements.txt` or `environment.yml`. |
| Medium | Move reusable prompt and plotting logic into `src/`. |

## One-Sentence Summary

AirPanel is a research workspace for evaluating synthetic consumer interviews about Orange and Bouygues ads, using LLM annotations, score modelling, lexical analysis, embeddings, demographic faceting, and KL-based sample-size stability analysis.
