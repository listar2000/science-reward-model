import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("dataset_building")
logging.basicConfig(level=logging.INFO)

SCORE_COLUMNS = ["novelty", "probability", "feasibility"]
REQUIRED_COLUMNS = [
    "context_puzzle",
    "title",
    "idea",
    "field_final",
    "author_perspective_based_on_ideas",
] + SCORE_COLUMNS
ALL_FIELDS = ["biology", "chemistry", "medicine", "other", "social"]

OUTPUT_LABELS = {
    "left": "A",
    "right": "B",
    "tie": "tie",
}

criteria_definitions = {
    "novelty": (
        "Evaluate the extent to which the hypotheses, generated based on the given context, introduce new ideas beyond the context."
    ),
    "feasibility": (
        "Evaluate the extent to which the hypotheses, generated based on the given context, can be feasibly tested, measured, or empirically investigated."
    ),
    "probability": (
        "Evaluate the extent to which the hypotheses, generated based on the given context, appear to be true - logically coherent, grounded in existing knowledge, and seemingly valid."
    ),
}

USER_PROMPT_TEMPLATE = """
You are an experienced scientist evaluating scientific hypotheses.

You will be given:
1. the **title** of a research paper
2. the **context**: background information and the research puzzle of the paper
3. two proposed **hypotheses**

Your task:
Compare the two hypotheses on three dimensions: novelty, feasibility, and probability of being true, according to the following criteria:
{criteria_definitions}

Title: {title}
Context: {context}
Your own perspective on this context can be summarized as the following hypotheses:
{perspectives_based_on_author_idea}

Hypothesis A:
{idea_A}

Hypothesis B:
{idea_B}

Return your answer as valid JSON with exactly these keys:
{{
  "novelty": "A" | "B" | "tie",
  "feasibility": "A" | "B" | "tie",
  "probability": "A" | "B" | "tie"
}}

Use:
- "A" if Hypothesis A is better on that dimension
- "B" if Hypothesis B is better on that dimension
- "tie" if they are tied on that dimension
"""


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before_len = len(df)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Column {col} not found in the data"
    df = df.dropna()
    after_len = len(df)
    print(
        f"Loaded {before_len} rows. Dropped {before_len - after_len} rows. Remaining {after_len} rows."
    )
    return df


def construct_messages(
    title: str,
    context: str,
    perspectives: str,
    idea_A: str,
    idea_B: str,
    allow_tie: bool,
) -> list[dict[str, str]]:
    content = USER_PROMPT_TEMPLATE.format(
        criteria_definitions=criteria_definitions,
        title=title,
        context=context,
        perspectives_based_on_author_idea=perspectives,
        idea_A=idea_A,
        idea_B=idea_B,
    )
    if not allow_tie:
        content = content.replace('"A" | "B" | "tie"', '"A" | "B"')
        content = content.replace('- "tie" if they are tied on that dimension\n', '')


    return [{"role": "user", "content": content.strip()}]


def pair_to_labels(
    first_scores: dict[str, float],
    second_scores: dict[str, float],
    allow_tie: bool,
) -> dict[str, str] | None:
    labels: dict[str, str] = {}
    label_map = OUTPUT_LABELS
    for score_col in SCORE_COLUMNS:
        first_score = first_scores[score_col]
        second_score = second_scores[score_col]
        if first_score > second_score:
            winner = "left"
        elif first_score < second_score:
            winner = "right"
        else:
            if not allow_tie:
                return None
            winner = "tie"
        labels[score_col] = label_map[winner]
    return labels


def reverse_labels(labels: dict[str, str]) -> dict[str, str]:
    reversed_labels = {"A": "B", "B": "A", "tie": "tie"}
    return {key: reversed_labels[value] for key, value in labels.items()}


def build_single_row(
    first_row: pd.Series,
    group_id: str,
    idea_A: str,
    idea_B: str,
    labels: dict[str, str],
    allow_tie: bool,
) -> dict[str, object]:
    messages = construct_messages(
        first_row["title"],
        first_row["context_puzzle"],
        first_row["author_perspective_based_on_ideas"],
        idea_A,
        idea_B,
        allow_tie,
    ) + [{"role": "assistant", "content": json.dumps(labels)}]

    return {
        "id": group_id,
        "messages": messages,
        "idea_A": idea_A,
        "idea_B": idea_B,
        **{f"label_{k}": v for k, v in labels.items()},
    }


def build_output_rows(
    group_df: pd.DataFrame,
    first_idx,
    second_idx,
    allow_tie: bool,
    add_reversed_pairs: bool,
) -> pd.DataFrame:
    first_row = group_df.iloc[0]
    group_id = first_row["id"]
    ideas = group_df["idea"].to_numpy()
    scores = {c: group_df[c].to_numpy() for c in SCORE_COLUMNS}

    rows = []
    for left_idx, right_idx in zip(first_idx, second_idx, strict=False):
        labels = pair_to_labels(
            {score_col: scores[score_col][left_idx] for score_col in SCORE_COLUMNS},
            {score_col: scores[score_col][right_idx] for score_col in SCORE_COLUMNS},
            allow_tie=allow_tie,
        )
        if labels is None:
            continue
        rows.append(
            build_single_row(
                first_row=first_row,
                group_id=group_id,
                idea_A=ideas[left_idx],
                idea_B=ideas[right_idx],
                labels=labels,
                allow_tie=allow_tie,
            )
        )
        if add_reversed_pairs:
            rows.append(
                build_single_row(
                    first_row=first_row,
                    group_id=group_id,
                    idea_A=ideas[right_idx],
                    idea_B=ideas[left_idx],
                    labels=reverse_labels(labels),
                    allow_tie=allow_tie,
                )
            )

    return pd.DataFrame(rows)


def per_group_construction(
    group_df: pd.DataFrame, allow_tie: bool, add_reversed_pairs: bool
) -> pd.DataFrame:
    if len(group_df) < 2:
        return pd.DataFrame()

    first_idx, second_idx = np.triu_indices(len(group_df), k=1)
    return build_output_rows(
        group_df, first_idx, second_idx, allow_tie, add_reversed_pairs
    )


def per_group_construction_adjacent(
    group_df: pd.DataFrame, allow_tie: bool, add_reversed_pairs: bool
) -> pd.DataFrame:
    """
    Sort the responses by their average human score, then only create adjacent pairs.
    """
    if len(group_df) < 2:
        return pd.DataFrame()

    group_df = group_df.copy()
    group_df["avg_score"] = group_df[SCORE_COLUMNS].mean(axis=1)
    group_df = group_df.sort_values("avg_score", ascending=False).reset_index(drop=True)

    m = len(group_df) - 1
    first_idx = np.arange(m)
    second_idx = np.arange(1, m + 1)
    return build_output_rows(
        group_df, first_idx, second_idx, allow_tie, add_reversed_pairs
    )


def per_group_construction_extreme(
    group_df: pd.DataFrame, allow_tie: bool, add_reversed_pairs: bool
) -> pd.DataFrame:
    """
    Sort responses by average human score, then only pair the best vs worst.
    """
    if len(group_df) < 2:
        return pd.DataFrame()

    group_df = group_df.copy()
    group_df["avg_score"] = group_df[SCORE_COLUMNS].mean(axis=1)
    group_df = group_df.sort_values("avg_score", ascending=False).reset_index(drop=True)

    return build_output_rows(
        group_df,
        np.array([0]),
        np.array([len(group_df) - 1]),
        allow_tie,
        add_reversed_pairs,
    )


def dataset_construction(
    raw_df: pd.DataFrame,
    pairing_strategy: str = "adjacent",
    allow_tie: bool = False,
    add_reversed_pairs: bool = False,
) -> pd.DataFrame:
    if pairing_strategy == "adjacent":
        constructor = per_group_construction_adjacent
    elif pairing_strategy == "extreme":
        constructor = per_group_construction_extreme
    else:
        constructor = per_group_construction

    parts = []
    for _, g in tqdm(
        raw_df.groupby("id", sort=False),
        desc=f"Constructing generative dataset ({pairing_strategy})",
    ):
        df_g = constructor(g, allow_tie, add_reversed_pairs)
        if not df_g.empty:
            parts.append(df_g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sample_ood_groups(
    raw_df: pd.DataFrame, field: str, n_groups: int, random_state: int
) -> pd.DataFrame:
    field_df = raw_df[raw_df["field_final"] == field]
    group_ids = sorted(field_df["id"].unique())
    n = min(n_groups, len(group_ids))
    rng = np.random.RandomState(random_state)
    sampled_ids = rng.choice(group_ids, size=n, replace=False)
    return raw_df[raw_df["id"].isin(sampled_ids)]


def main():
    parser = argparse.ArgumentParser(
        description="Build generative comparison dataset for one-vs-rest field generalisation."
    )
    parser.add_argument("--input", required=True, help="Path to raw CSV data")
    parser.add_argument(
        "--train_field",
        required=True,
        choices=ALL_FIELDS,
        help="Field to train on; the remaining 4 fields become OOD test sets",
    )
    parser.add_argument(
        "--n_ood_groups",
        type=int,
        default=100,
        help="Number of groups to sample per OOD field (default: 100). If a field has fewer groups, all are used.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of train-field groups held out as ID test (default: 0.1)",
    )
    parser.add_argument("--folder", required=True, help="Output folder for CSV files")
    parser.add_argument(
        "--pairing_strategy",
        default="adjacent",
        choices=["exhaustive", "adjacent", "extreme"],
        help="Pairing strategy (default: adjacent)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--allow_tie",
        action="store_true",
        help="Keep tied comparisons and emit \"tie\" in the JSON labels. By default, tied comparisons are skipped.",
    )
    parser.add_argument(
        "--add_reversed_pairs",
        action="store_true",
        help="Add a second copy of each pair with A/B swapped and labels reversed to reduce position bias.",
    )
    args = parser.parse_args()

    raw_df = load_raw_data(args.input)

    ood_fields = [f for f in ALL_FIELDS if f != args.train_field]

    train_field_df = raw_df[raw_df["field_final"] == args.train_field]
    all_train_group_ids = pd.Series(sorted(train_field_df["id"].unique()))
    n_id_test = int(len(all_train_group_ids) * args.test_ratio)
    id_test_ids = set(
        all_train_group_ids.sample(n=n_id_test, random_state=args.random_state).values
    )
    train_ids = set(all_train_group_ids.values) - id_test_ids

    logger.info(
        f"Train field '{args.train_field}': "
        f"{len(train_ids)} train groups, {len(id_test_ids)} ID test groups"
    )

    train_df = raw_df[raw_df["id"].isin(train_ids)]
    id_test_df = raw_df[raw_df["id"].isin(id_test_ids)]

    train_built = dataset_construction(
        train_df,
        args.pairing_strategy,
        allow_tie=args.allow_tie,
        add_reversed_pairs=args.add_reversed_pairs,
    )
    id_test_built = dataset_construction(
        id_test_df,
        args.pairing_strategy,
        allow_tie=args.allow_tie,
        add_reversed_pairs=args.add_reversed_pairs,
    )
    logger.info(
        f"Built {len(train_built)} train examples, {len(id_test_built)} ID test examples"
    )

    os.makedirs(args.folder, exist_ok=True)
    train_built.to_csv(os.path.join(args.folder, "built_train.csv"), index=False)
    id_test_built.to_csv(
        os.path.join(args.folder, f"built_ID_test_{args.train_field}.csv"), index=False
    )

    for field in ood_fields:
        ood_df = sample_ood_groups(raw_df, field, args.n_ood_groups, args.random_state)
        n_sampled = ood_df["id"].nunique()
        ood_built = dataset_construction(
            ood_df,
            args.pairing_strategy,
            allow_tie=args.allow_tie,
            add_reversed_pairs=args.add_reversed_pairs,
        )
        out_path = os.path.join(args.folder, f"built_OOD_test_{field}.csv")
        ood_built.to_csv(out_path, index=False)
        logger.info(
            f"OOD field '{field}': {n_sampled} groups -> {len(ood_built)} examples -> {out_path}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
