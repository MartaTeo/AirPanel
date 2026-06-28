from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "interviews.csv"

DEFAULT_MODEL_NAME = "mistral:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
TEMPERATURE = 0.1
TIMEOUT_SECONDS = 120
MAX_RETRIES = 2
CHECKPOINT_EVERY = 25
CONFIDENCE_THRESHOLD = 0.7

RAW_OUTPUT = BASE_DIR / "preference_annotations_question_scores.csv"
CHECKPOINT_OUTPUT = BASE_DIR / "annotations_question_scores_checkpoint.csv"
WIDE_OUTPUT = BASE_DIR / "interviews_with_question_scores.csv"
SUMMARY_OUTPUT = BASE_DIR / "question_score_summary.csv"
CODEBOOK_OUTPUT = BASE_DIR / "question_score_codebook.csv"
RUN_INFO_OUTPUT = BASE_DIR / "question_score_run.json"

QUESTION_COLUMNS = [
    "Q_bouygues_reactions_spontanees_1",
    "Q_bouygues_memorisation_1",
    "Q_bouygues_caractere_distinctif_1",
    "Q_bouygues_attractivite_1",
    "Q_bouygues_attractivite_2",
    "Q_bouygues_resonance_emotionnelle_1",
    "Q_bouygues_image_1",
    "Q_bouygues_intention_achat_1",
    "Q_orange_reactions_spontanees_1",
    "Q_orange_memorisation_1",
    "Q_orange_caractere_distinctif_1",
    "Q_orange_attractivite_1",
    "Q_orange_attractivite_2",
    "Q_orange_resonance_emotionnelle_1",
    "Q_orange_image_1",
    "Q_orange_intention_achat_1",
    "Q_comparaison_1",
    "Q_comparaison_2",
    "Q_comparaison_3",
    "Q_comparaison_4",
    "Q_comparaison_5",
    "Q_comparaison_6",
]

SCHEMA_SCORES = {
    "creativity": "Rate how much this answer contains evidence of creativity or creative execution for each brand.",
    "humour": "Rate how much this answer contains evidence of humour, amusement, or entertainment for each brand.",
    "originality": "Rate how much this answer contains evidence of originality, distinctiveness, or memorability for each brand.",
    "reliability_trust": "Rate how much this answer contains evidence of reliability, trust, reassurance, or credibility for each brand.",
    "intent_to_purchase": "Rate how much this answer contains evidence that the panelist would consider subscribing, switching, or finding out more for each brand.",
    "overall": "Rate the local overall positive evaluation or preference evidence for each brand in this answer only.",
}

BRANDS = ["Bouygues", "Orange"]
ALL_DIMS = list(SCHEMA_SCORES.keys())

FEW_SHOTS = [
    {
        "question_group": "Bouygues",
        "answer": "L'humour. Franchement, c'est drole. Ils ont pris un probleme de WiFi et ils en ont fait une scene de crime. C'est original et on s'en souvient.",
        "label": {
            "creativity": {"Bouygues": 8, "Orange": 0},
            "humour": {"Bouygues": 9, "Orange": 0},
            "originality": {"Bouygues": 8, "Orange": 0},
            "reliability_trust": {"Bouygues": 3, "Orange": 0},
            "intent_to_purchase": {"Bouygues": 4, "Orange": 0},
            "overall": {"Bouygues": 7, "Orange": 0},
            "reasoning": "The answer gives strong creative and humorous evidence for Bouygues only.",
            "confidence": 0.95,
        },
    },
    {
        "question_group": "Orange",
        "answer": "Le message de la fibre la plus fiable me parle. C'est simple, concret, rassurant, et c'est ce qui me donnerait envie de me renseigner.",
        "label": {
            "creativity": {"Bouygues": 0, "Orange": 3},
            "humour": {"Bouygues": 0, "Orange": 1},
            "originality": {"Bouygues": 0, "Orange": 3},
            "reliability_trust": {"Bouygues": 0, "Orange": 9},
            "intent_to_purchase": {"Bouygues": 0, "Orange": 8},
            "overall": {"Bouygues": 0, "Orange": 8},
            "reasoning": "The answer gives strong reliability and purchase-intent evidence for Orange only.",
            "confidence": 0.95,
        },
    },
    {
        "question_group": "Comparison",
        "answer": "Je prefere Bouygues pour l'originalite et l'humour, mais Orange est plus rassurant et me convainc davantage sur la fiabilite.",
        "label": {
            "creativity": {"Bouygues": 8, "Orange": 4},
            "humour": {"Bouygues": 8, "Orange": 2},
            "originality": {"Bouygues": 9, "Orange": 4},
            "reliability_trust": {"Bouygues": 4, "Orange": 9},
            "intent_to_purchase": {"Bouygues": 5, "Orange": 7},
            "overall": {"Bouygues": 7, "Orange": 7},
            "reasoning": "The answer compares both brands and assigns different strengths by dimension.",
            "confidence": 0.95,
        },
    },
]


def question_group(question_name: str) -> str:
    if question_name.startswith("Q_bouygues_"):
        return "Bouygues"
    if question_name.startswith("Q_orange_"):
        return "Orange"
    return "Comparison"


def question_family(question_name: str) -> str:
    if question_name.startswith("Q_bouygues_"):
        return question_name.replace("Q_bouygues_", "")
    if question_name.startswith("Q_orange_"):
        return question_name.replace("Q_orange_", "")
    return question_name.replace("Q_comparaison_", "comparaison_")


def build_codebook() -> pd.DataFrame:
    rows = []
    for i, question_name in enumerate(QUESTION_COLUMNS, start=1):
        rows.append(
            {
                "question_id": f"q{i}",
                "question_number": i,
                "question_column": question_name,
                "question_group": question_group(question_name),
                "question_family": question_family(question_name),
            }
        )
    return pd.DataFrame(rows)


def build_long_question_table(panel: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in QUESTION_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing expected question columns in {INPUT_FILE}: {missing}")

    codebook = build_codebook()
    records = []
    for _, row in panel.iterrows():
        panelist_id = row["panelist_id"]
        for _, q in codebook.iterrows():
            text = row[q["question_column"]]
            if pd.isna(text) or str(text).strip() == "":
                continue
            records.append(
                {
                    "panelist_id": panelist_id,
                    "question_id": q["question_id"],
                    "question_number": int(q["question_number"]),
                    "question_column": q["question_column"],
                    "question_group": q["question_group"],
                    "question_family": q["question_family"],
                    "answer_text": str(text),
                }
            )
    return pd.DataFrame(records)


def extract_json(text: str) -> dict | None:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else None
    except Exception:
        return None


def normalize_score(value) -> int:
    try:
        return max(0, min(10, int(round(float(value)))))
    except Exception:
        return -1


def validate_scores(parsed: dict) -> dict:
    for dim in ALL_DIMS:
        if dim not in parsed or not isinstance(parsed[dim], dict):
            parsed[dim] = {"Bouygues": -1, "Orange": -1}
        for brand in BRANDS:
            parsed[dim][brand] = normalize_score(parsed[dim].get(brand, -1))
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0
    parsed["reasoning"] = str(parsed.get("reasoning", ""))[:500]
    return parsed


def build_system_prompt() -> str:
    schema_lines = "\n".join(f"- {dim}: {rule}" for dim, rule in SCHEMA_SCORES.items())
    field_lines = "\n".join(f'  "{dim}": {{"Bouygues": <0-10>, "Orange": <0-10>}},' for dim in ALL_DIMS)
    return f"""You are an expert annotator for advertising discourse analysis.
You score one answer from one panelist who watched two TV advertisements:
- Bouygues Telecom
- Orange

The answer may be about Bouygues only, Orange only, or a direct comparison.
Score ONLY the evidence present in this single answer.
Do not use other answers from the same panelist.
For a brand-specific answer, the non-discussed brand should usually receive 0 unless it is explicitly mentioned.

Scores are integers from 0 to 10:
0 = no evidence in this answer
10 = extremely strong evidence in this answer

DIMENSIONS:
{schema_lines}

STRICT RULES:
1. Return ONLY one valid JSON object.
2. Each dimension must contain both brands.
3. Scores must be integers between 0 and 10.
4. Add a short "reasoning" field.
5. Add a "confidence" field between 0.0 and 1.0.

OUTPUT FORMAT:
{{
{field_lines}
  "reasoning": "short explanation",
  "confidence": 0.95
}}"""


SYSTEM_PROMPT = build_system_prompt()


def build_user_prompt(row: pd.Series) -> str:
    prompt = "ANNOTATED EXAMPLES:\n"
    for ex in FEW_SHOTS:
        prompt += f"Question group: {ex['question_group']}\n"
        prompt += f"Answer: {ex['answer']}\n"
        prompt += f"Annotation: {json.dumps(ex['label'], ensure_ascii=False)}\n---\n"

    prompt += "ANSWER TO SCORE:\n"
    prompt += f"Panelist id: {row['panelist_id']}\n"
    prompt += f"Question id: {row['question_id']}\n"
    prompt += f"Question column: {row['question_column']}\n"
    prompt += f"Question group: {row['question_group']}\n"
    prompt += f"Question family: {row['question_family']}\n"
    prompt += f"Answer: {row['answer_text']}\n"
    return prompt


def available_ollama_models() -> set[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
        return {m.get("name", "") for m in payload.get("models", [])}
    except Exception:
        return set()


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def output_paths(run_name: str | None = None) -> dict[str, Path]:
    if not run_name:
        return {
            "raw": RAW_OUTPUT,
            "checkpoint": CHECKPOINT_OUTPUT,
            "wide": WIDE_OUTPUT,
            "summary": SUMMARY_OUTPUT,
            "codebook": CODEBOOK_OUTPUT,
            "run_info": RUN_INFO_OUTPUT,
        }

    slug = safe_slug(run_name)
    return {
        "raw": BASE_DIR / f"preference_annotations_question_scores_{slug}.csv",
        "checkpoint": BASE_DIR / f"annotations_question_scores_checkpoint_{slug}.csv",
        "wide": BASE_DIR / f"interviews_with_question_scores_{slug}.csv",
        "summary": BASE_DIR / f"question_score_summary_{slug}.csv",
        "codebook": BASE_DIR / f"question_score_codebook_{slug}.csv",
        "run_info": BASE_DIR / f"question_score_run_{slug}.json",
    }


def choose_model(requested_model: str) -> str:
    models = available_ollama_models()
    if not models:
        raise RuntimeError(
            "Ollama is not reachable or no local models are installed. "
            "Start Ollama and make sure mistral:7b or mistral is available before running this script."
        )
    if requested_model in models:
        return requested_model
    matching = [name for name in models if name.startswith(requested_model)]
    if matching:
        return sorted(matching)[0]
    if requested_model == DEFAULT_MODEL_NAME and "mistral" in models:
        return "mistral"
    raise RuntimeError(
        f"Expected {requested_model} in Ollama. Available models: {sorted(models)}"
    )


def annotate(row: pd.Series, model_name: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": TEMPERATURE},
                },
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            generated = response.json()["message"]["content"]
            parsed = extract_json(generated)
            if parsed is None:
                out = {dim: {"Bouygues": -1, "Orange": -1} for dim in ALL_DIMS}
                out["reasoning"] = generated[:500]
                out["confidence"] = 0.0
                out["raw_output"] = generated
                return out
            parsed = validate_scores(parsed)
            parsed["raw_output"] = generated
            return parsed
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 + attempt)

    out = {dim: {"Bouygues": -1, "Orange": -1} for dim in ALL_DIMS}
    out["reasoning"] = f"AnnotationError: {last_error}"[:500]
    out["confidence"] = 0.0
    out["raw_output"] = last_error
    return out


def load_completed(paths: dict[str, Path]) -> pd.DataFrame:
    if paths["checkpoint"].exists():
        return pd.read_csv(paths["checkpoint"])
    if paths["raw"].exists():
        return pd.read_csv(paths["raw"])
    return pd.DataFrame()


def completed_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df["panelist_id"].astype(str), df["question_id"].astype(str)))


def annotation_row(base_row: pd.Series, result: dict, model_name: str) -> dict:
    row = {
        "panelist_id": base_row["panelist_id"],
        "question_id": base_row["question_id"],
        "question_number": base_row["question_number"],
        "question_column": base_row["question_column"],
        "question_group": base_row["question_group"],
        "question_family": base_row["question_family"],
        "answer_text": base_row["answer_text"],
        "model": model_name,
        "temperature": TEMPERATURE,
    }
    for dim in ALL_DIMS:
        row[dim] = result[dim]
        row[f"{dim}_flagged"] = result.get("confidence", 0.0) < CONFIDENCE_THRESHOLD or any(
            result[dim].get(brand, -1) < 0 for brand in BRANDS
        )
    row["reasoning"] = result.get("reasoning", "")
    row["confidence"] = result.get("confidence", 0.0)
    row["raw_output"] = result.get("raw_output", "")
    return row


def save_annotations(rows: list[dict], paths: dict[str, Path]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values(["panelist_id", "question_number"]).reset_index(drop=True)
    df.to_csv(paths["checkpoint"], index=False)
    return df


def parse_score_cell(value) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {"Bouygues": -1, "Orange": -1}
    text = str(value)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"Bouygues": -1, "Orange": -1}


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def build_wide_scores(panel: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    score_rows = []
    for panelist_id, part in annotations.groupby("panelist_id", sort=False):
        row = {"panelist_id": panelist_id}
        for _, ann in part.iterrows():
            qid = ann["question_id"]
            row[f"question_column_{qid}"] = ann["question_column"]
            row[f"confidence_{qid}"] = ann["confidence"]
            row[f"reasoning_{qid}"] = ann["reasoning"]
            for dim in ALL_DIMS:
                scores = parse_score_cell(ann[dim])
                for brand in BRANDS:
                    row[f"{dim}_{qid}_{brand.lower()}"] = normalize_score(scores.get(brand, -1))
        score_rows.append(row)

    scores_wide = pd.DataFrame(score_rows)
    return panel.merge(scores_wide, on="panelist_id", how="left")


def build_summary(annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ann in annotations.iterrows():
        for dim in ALL_DIMS:
            scores = parse_score_cell(ann[dim])
            for brand in BRANDS:
                rows.append(
                    {
                        "question_id": ann["question_id"],
                        "question_number": ann["question_number"],
                        "question_column": ann["question_column"],
                        "question_group": ann["question_group"],
                        "question_family": ann["question_family"],
                        "dimension": dim,
                        "brand": brand,
                        "score": normalize_score(scores.get(brand, -1)),
                        "confidence": ann["confidence"],
                        "flagged": parse_bool(ann.get(f"{dim}_flagged", False)),
                    }
                )
    long_scores = pd.DataFrame(rows)
    long_scores = long_scores[long_scores["score"] >= 0].copy()
    return (
        long_scores.groupby(
            ["question_id", "question_number", "question_column", "question_group", "question_family", "dimension", "brand"],
            as_index=False,
        )
        .agg(
            n=("score", "size"),
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            median_score=("score", "median"),
            mean_confidence=("confidence", "mean"),
            flagged_rate=("flagged", "mean"),
        )
        .sort_values(["question_number", "dimension", "brand"])
    )


def write_run_info(model_name: str, n_tasks: int, n_completed: int, n_rows: int | None, paths: dict[str, Path]) -> None:
    payload = {
        "input": str(INPUT_FILE),
        "raw_output": str(paths["raw"]),
        "checkpoint_output": str(paths["checkpoint"]),
        "wide_output": str(paths["wide"]),
        "summary_output": str(paths["summary"]),
        "codebook_output": str(paths["codebook"]),
        "model": model_name,
        "temperature": TEMPERATURE,
        "n_rows": n_rows,
        "dimensions": ALL_DIMS,
        "brands": BRANDS,
        "question_columns": QUESTION_COLUMNS,
        "n_tasks": n_tasks,
        "n_completed": n_completed,
        "interpretation": "Scores are per-question evidence scores. They are not the same as final panelist-level preference scores.",
    }
    paths["run_info"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score each AirPanel question answer by dimension and brand using a local Ollama model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Ollama model name, e.g. mistral:7b, mistral, gemma3, gemma4.")
    parser.add_argument("--n_rows", type=int, default=None, help="Only score the first N panelists from interviews.csv.")
    parser.add_argument("--run_name", default=None, help="Optional suffix for output files, useful for test runs or model comparisons.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    panel = pd.read_csv(INPUT_FILE)
    if args.n_rows is not None:
        if args.n_rows <= 0:
            raise ValueError("--n_rows must be a positive integer")
        panel = panel.head(args.n_rows).copy()

    run_name = args.run_name
    if run_name is None and (args.n_rows is not None or args.model != DEFAULT_MODEL_NAME):
        n_part = f"first{args.n_rows}" if args.n_rows is not None else "all"
        run_name = f"{safe_slug(args.model)}_{n_part}"
    paths = output_paths(run_name)

    codebook = build_codebook()
    codebook.to_csv(paths["codebook"], index=False)

    tasks = build_long_question_table(panel)
    existing = load_completed(paths)
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(existing)
    model_name = choose_model(args.model)

    remaining = tasks[
        ~tasks.apply(lambda r: (str(r["panelist_id"]), str(r["question_id"])) in done, axis=1)
    ].copy()

    print(f"Input interviews: {len(panel)}")
    print(f"Question scoring tasks: {len(tasks)}")
    print(f"Already completed: {len(done)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Model: {model_name}")
    print(f"Output run name: {run_name or 'default'}")
    print(f"Checkpoint: {paths['checkpoint']}")

    start = time.time()
    for pos, (_, task) in enumerate(remaining.iterrows(), start=1):
        result = annotate(task, model_name)
        rows.append(annotation_row(task, result, model_name))

        if pos % CHECKPOINT_EVERY == 0 or pos == len(remaining):
            annotations = save_annotations(rows, paths)
            elapsed = time.time() - start
            avg = elapsed / max(1, pos)
            eta = avg * (len(remaining) - pos) / 60
            print(
                f"{len(done) + pos}/{len(tasks)} saved | "
                f"{avg:.1f}s/task | ETA {eta:.1f} min | "
                f"last={task['panelist_id']} {task['question_id']}"
            )

    annotations = save_annotations(rows, paths)
    annotations.to_csv(paths["raw"], index=False)

    wide = build_wide_scores(panel, annotations)
    wide.to_csv(paths["wide"], index=False)

    summary = build_summary(annotations)
    summary.to_csv(paths["summary"], index=False)

    write_run_info(model_name, len(tasks), len(annotations), args.n_rows, paths)

    print("")
    print(f"Saved raw long annotations: {paths['raw']}")
    print(f"Saved wide panel dataset: {paths['wide']}")
    print(f"Saved summary: {paths['summary']}")
    print(f"Saved codebook: {paths['codebook']}")
    print(f"Saved run info: {paths['run_info']}")
    print(f"Wide shape: {wide.shape}")


if __name__ == "__main__":
    main()
