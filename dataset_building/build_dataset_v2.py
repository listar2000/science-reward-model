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
    "field_fine",
    "author_perspective_based_on_ideas",
] + SCORE_COLUMNS

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
    group_rater = first_row["rater"]
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
        "rater": np.repeat(group_rater, m),
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
    group_rater = first_row["rater"]
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
        "rater": np.repeat(group_rater, m),
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
    group_rater = best_row["rater"]
    group_user_message = construct_user_message(
        best_row["title"], best_row["context_puzzle"]
    )

    out = {
        "id": [group_id],
        "rater": [group_rater],
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


def field_aware_train_test_split(
    group_ids: pd.Series,
    group_fields: pd.Series,
    train_fields: list[str],
    test_fields: list[str],
    test_ratio: float,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split group IDs into train, OOD test, and ID test sets.

    Args:
        group_ids: Series of group-ID strings (one entry per group).
        group_fields: Series of field_fine values aligned with group_ids.
        train_fields: field_fine values whose groups are eligible for train/ID-test.
        test_fields: field_fine values whose groups are forced into OOD test.
        test_ratio: Fraction of train-field groups to hold out as ID test.
        random_state: Seed for reproducible sampling.

    Returns:
        (train_ids, forced_test_ids, extra_test_ids)
    """
    # OOD test: all groups whose field_fine is in test_fields
    ood_mask = group_fields.isin(test_fields)
    forced_test_idx = group_ids.index[ood_mask]

    # Train pool: all groups whose field_fine is in train_fields
    train_mask = group_fields.isin(train_fields)
    train_pool_ids = group_ids.loc[train_mask]

    # ID test: test_ratio fraction sampled from the train pool
    n_id_test = int(len(train_pool_ids) * test_ratio)
    if n_id_test > 0:
        extra_test_idx = train_pool_ids.sample(
            n=n_id_test, random_state=random_state
        ).index
    else:
        extra_test_idx = pd.Index([])

    all_test_idx = forced_test_idx.append(extra_test_idx)
    train_ids = group_ids.drop(all_test_idx).reset_index(drop=True)
    forced_test_ids = group_ids.loc[forced_test_idx].reset_index(drop=True)
    extra_test_ids = group_ids.loc[extra_test_idx].reset_index(drop=True)

    return train_ids, forced_test_ids, extra_test_ids


def main():
    parser = argparse.ArgumentParser(description="Build pairwise preference dataset")
    parser.add_argument("--input", required=True, help="Path to raw CSV data")
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="field_fine values whose groups are used for training (and ID test sampling)",
    )
    parser.add_argument(
        "--test",
        nargs="+",
        required=True,
        help="field_fine values whose groups are forced into OOD test",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        required=True,
        help="Fraction of train-field groups held out as ID test",
    )
    parser.add_argument(
        "--folder", required=True, help="Output folder for the 3 CSV files"
    )
    parser.add_argument(
        "--pairing_strategy",
        default="adjacent",
        choices=["exhaustive", "adjacent", "extreme"],
        help="Pairing strategy (default: adjacent)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    raw_df = load_raw_data(args.input)

    # One row per group for splitting
    group_info = (
        raw_df.groupby("id", sort=False).first().reset_index()[["id", "field_fine"]]
    )
    group_ids = pd.Series(group_info["id"].values, index=group_info.index)
    group_fields = pd.Series(group_info["field_fine"].values, index=group_info.index)

    logger.info(f"Loaded {len(raw_df)} ideas from {len(group_ids)} groups")

    train_ids, forced_test_ids, extra_test_ids = field_aware_train_test_split(
        group_ids,
        group_fields,
        args.train,
        args.test,
        args.test_ratio,
        args.random_state,
    )
    logger.info(
        f"Split into {len(train_ids)} train groups, "
        f"{len(forced_test_ids)} OOD test groups, "
        f"{len(extra_test_ids)} ID test groups"
    )

    train_df = raw_df[raw_df["id"].isin(train_ids)]
    forced_test_df = raw_df[raw_df["id"].isin(forced_test_ids)]
    extra_test_df = raw_df[raw_df["id"].isin(extra_test_ids)]

    train_built = dataset_construction(train_df, args.pairing_strategy)
    forced_test_built = dataset_construction(forced_test_df, args.pairing_strategy)
    extra_test_built = dataset_construction(extra_test_df, args.pairing_strategy)
    logger.info(
        f"Built {len(train_built)} train rows, "
        f"{len(forced_test_built)} OOD test rows, "
        f"{len(extra_test_built)} ID test rows"
    )

    os.makedirs(args.folder, exist_ok=True)
    train_built.to_csv(os.path.join(args.folder, "built_train.csv"), index=False)
    forced_test_built.to_csv(
        os.path.join(args.folder, "built_forced_test.csv"), index=False
    )
    extra_test_built.to_csv(
        os.path.join(args.folder, "built_extra_test.csv"), index=False
    )
    logger.info(f"Saved 3 CSV files to {args.folder}/")


if __name__ == "__main__":
    main()
