import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

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

FIRST_IDEA_SUFFIX, SECOND_IDEA_SUFFIX = "left", "right"


criteria_definitions = {
    "novelty": (
        "Evaluate the extent to which the hypotheses, generated based on the given context, introduce new ideas beyond the context."
    ),
    "feasibility": (
        "Evaluate the extent to which the hypotheses, generated based on the given context, can be feasibly tested, measured, or empirically investigated."
    ),
    "probability of being true": (
        "Evaluate the extent to which the hypotheses, generated based on the given context, appear to be true - logically coherent, grounded in existing knowledge, and seemingly valid."
    ),
}

USER_PROMPT_TEMPLATE = """
You are an experienced scientist evaluating a scientific hypothesis.

  You will be given:
  1. the **title** of a research paper
  2. the **context**: background information and the research puzzle of the paper
  3. a proposed **hypothesis**

  Your task:
  Evaluate the hypothesis on three dimensions: novelty, feasibility, and probability of being true, according to the following criteria:
  {criteria_definitions}
  Title: {title}
  Context: {context}
  Your own perspective on this context can be summarized as the following hypotheses:
  {perspectives_based_on_author_idea}
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


def construct_user_message(
    title: str, context: str, perspectives: str
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                criteria_definitions=criteria_definitions,
                title=title,
                context=context,
                perspectives_based_on_author_idea=perspectives,
            ),
        }
    ]


def construct_assistant_message(idea: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": idea}]


def per_group_construction(group_df: pd.DataFrame) -> pd.DataFrame:
    first_row = group_df.iloc[0]
    group_id = first_row["id"]
    group_user_message = construct_user_message(
        first_row["title"],
        first_row["context_puzzle"],
        first_row["author_perspective_based_on_ideas"],
    )

    if len(group_df) < 2:
        return pd.DataFrame()

    ideas = group_df["idea"].to_numpy()
    scores = {c: group_df[c].to_numpy() for c in SCORE_COLUMNS}

    first_idx, second_idx = np.triu_indices(len(group_df), k=1)
    m = len(first_idx)

    out = {
        "id": np.repeat(group_id, m),
        "user_message": [group_user_message] * m,
        f"assistant_message_{FIRST_IDEA_SUFFIX}": [
            construct_assistant_message(idea) for idea in ideas[first_idx]
        ],
        f"assistant_message_{SECOND_IDEA_SUFFIX}": [
            construct_assistant_message(idea) for idea in ideas[second_idx]
        ],
    }

    for score_col in SCORE_COLUMNS:
        first_scores, second_scores = (
            scores[score_col][first_idx],
            scores[score_col][second_idx],
        )
        out[f"{score_col}_{FIRST_IDEA_SUFFIX}"] = first_scores
        out[f"{score_col}_{SECOND_IDEA_SUFFIX}"] = second_scores
        winners = np.where(
            first_scores > second_scores,
            FIRST_IDEA_SUFFIX,
            np.where(first_scores < second_scores, SECOND_IDEA_SUFFIX, "tie"),
        )
        out[f"winner_{score_col}"] = winners

    return pd.DataFrame(out)


def per_group_construction_adjacent(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the responses by their average human score (across novelty, probability, feasibility),
    then only create pairs between adjacent ranks: Rank 1 vs 2, Rank 2 vs 3, etc.
    """
    if len(group_df) < 2:
        return pd.DataFrame()

    group_df = group_df.copy()
    group_df["avg_score"] = group_df[SCORE_COLUMNS].mean(axis=1)
    group_df = group_df.sort_values("avg_score", ascending=False).reset_index(drop=True)

    first_row = group_df.iloc[0]
    group_id = first_row["id"]
    group_user_message = construct_user_message(
        first_row["title"],
        first_row["context_puzzle"],
        first_row["author_perspective_based_on_ideas"],
    )

    ideas = group_df["idea"].to_numpy()
    scores = {c: group_df[c].to_numpy() for c in SCORE_COLUMNS}
    m = len(group_df) - 1

    first_idx = np.arange(m)
    second_idx = np.arange(1, m + 1)

    out = {
        "id": np.repeat(group_id, m),
        "user_message": [group_user_message] * m,
        f"assistant_message_{FIRST_IDEA_SUFFIX}": [
            construct_assistant_message(idea) for idea in ideas[first_idx]
        ],
        f"assistant_message_{SECOND_IDEA_SUFFIX}": [
            construct_assistant_message(idea) for idea in ideas[second_idx]
        ],
    }

    for score_col in SCORE_COLUMNS:
        first_scores = scores[score_col][first_idx]
        second_scores = scores[score_col][second_idx]
        out[f"{score_col}_{FIRST_IDEA_SUFFIX}"] = first_scores
        out[f"{score_col}_{SECOND_IDEA_SUFFIX}"] = second_scores
        winners = np.where(
            first_scores > second_scores,
            FIRST_IDEA_SUFFIX,
            np.where(first_scores < second_scores, SECOND_IDEA_SUFFIX, "tie"),
        )
        out[f"winner_{score_col}"] = winners

    return pd.DataFrame(out)


def per_group_construction_extreme(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort responses by average human score, then only pair the best (rank 1) vs worst (rank N).
    Produces exactly 1 pair per group.
    """
    if len(group_df) < 2:
        return pd.DataFrame()

    group_df = group_df.copy()
    group_df["avg_score"] = group_df[SCORE_COLUMNS].mean(axis=1)
    group_df = group_df.sort_values("avg_score", ascending=False).reset_index(drop=True)

    best_row = group_df.iloc[0]
    worst_row = group_df.iloc[-1]

    group_id = best_row["id"]
    group_user_message = construct_user_message(
        best_row["title"],
        best_row["context_puzzle"],
        best_row["author_perspective_based_on_ideas"],
    )

    out = {
        "id": [group_id],
        "user_message": [group_user_message],
        f"assistant_message_{FIRST_IDEA_SUFFIX}": [
            construct_assistant_message(best_row["idea"])
        ],
        f"assistant_message_{SECOND_IDEA_SUFFIX}": [
            construct_assistant_message(worst_row["idea"])
        ],
    }

    for score_col in SCORE_COLUMNS:
        first_score = best_row[score_col]
        second_score = worst_row[score_col]
        out[f"{score_col}_{FIRST_IDEA_SUFFIX}"] = [first_score]
        out[f"{score_col}_{SECOND_IDEA_SUFFIX}"] = [second_score]
        if first_score > second_score:
            winner = FIRST_IDEA_SUFFIX
        elif first_score < second_score:
            winner = SECOND_IDEA_SUFFIX
        else:
            winner = "tie"
        out[f"winner_{score_col}"] = [winner]

    return pd.DataFrame(out)


def dataset_construction(
    raw_df: pd.DataFrame, pairing_strategy: str = "adjacent"
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
        desc=f"Constructing dataset ({pairing_strategy})",
    ):
        df_g = constructor(g)
        if not df_g.empty:
            parts.append(df_g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sample_ood_groups(
    raw_df: pd.DataFrame, field: str, n_groups: int, random_state: int
) -> pd.DataFrame:
    """
    Sample up to n_groups groups from the given field using a fixed seed.

    The seed is applied to the sorted list of group IDs, so the same groups
    are always selected for a given field regardless of which field is being
    trained on. If the field has fewer than n_groups groups, all are used.
    """
    field_df = raw_df[raw_df["field_final"] == field]
    group_ids = sorted(field_df["id"].unique())  # sort for determinism
    n = min(n_groups, len(group_ids))
    rng = np.random.RandomState(random_state)
    sampled_ids = rng.choice(group_ids, size=n, replace=False)
    return raw_df[raw_df["id"].isin(sampled_ids)]


def main():
    parser = argparse.ArgumentParser(
        description="Build pairwise preference dataset for one-vs-rest field generalisation."
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
        help="Number of groups to sample per OOD field (default: 100). "
        "If a field has fewer groups, all are used.",
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
    args = parser.parse_args()

    raw_df = load_raw_data(args.input)

    ood_fields = [f for f in ALL_FIELDS if f != args.train_field]

    # --- Train field: split into train and ID test ---
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

    # --- Build train and ID test datasets ---
    train_built = dataset_construction(train_df, args.pairing_strategy)
    id_test_built = dataset_construction(id_test_df, args.pairing_strategy)
    logger.info(
        f"Built {len(train_built)} train pairs, {len(id_test_built)} ID test pairs"
    )

    os.makedirs(args.folder, exist_ok=True)
    train_built.to_csv(os.path.join(args.folder, "built_train.csv"), index=False)
    id_test_built.to_csv(
        os.path.join(args.folder, f"built_ID_test_{args.train_field}.csv"), index=False
    )

    # --- OOD test: one file per field, sampled with fixed seed ---
    for field in ood_fields:
        ood_df = sample_ood_groups(raw_df, field, args.n_ood_groups, args.random_state)
        n_sampled = ood_df["id"].nunique()
        ood_built = dataset_construction(ood_df, args.pairing_strategy)
        out_path = os.path.join(args.folder, f"built_OOD_test_{field}.csv")
        ood_built.to_csv(out_path, index=False)
        logger.info(
            f"OOD field '{field}': {n_sampled} groups → {len(ood_built)} pairs → {out_path}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
