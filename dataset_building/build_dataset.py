import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import logging

from filter_utils import category_aware_train_test_split
from hf_utils import upload_splits_to_hub

logger = logging.getLogger("dataset_building")

SCORE_COLUMNS = ["novelty", "probability", "feasibility"]
REQUIRED_COLUMNS = ["context_puzzle", "title", "idea"] + SCORE_COLUMNS

FIRST_IDEA_SUFFIX, SECOND_IDEA_SUFFIX = "left", "right"


def load_raw_data(cfg: DictConfig) -> pd.DataFrame:
    path = cfg.input.raw_data_path
    df = pd.read_csv(path)
    before_len = len(df)
    # make sure the title and context columns are present
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Column {col} not found in the data"
    df = df.dropna()
    after_len = len(df)
    print(
        f"Loaded {before_len} rows. Dropped {before_len - after_len} rows. Remaining {after_len} rows."
    )
    return df


def output_path_helper(filename: str, cfg: DictConfig) -> str:
    path = os.path.join(cfg.output.output_data_folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def construct_user_message(
    title: str, context: str, cfg: DictConfig
) -> list[dict[str, str]]:
    template = cfg.user_prompt_template
    return [{"role": "user", "content": template.format(title=title, context=context)}]


def construct_assistant_message(idea: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": idea}]


def per_pair_construction(first_row, second_row, cfg: DictConfig) -> dict:
    pair = {
        f"assistant_message_{FIRST_IDEA_SUFFIX}": construct_assistant_message(
            first_row["idea"]
        ),
        f"assistant_message_{SECOND_IDEA_SUFFIX}": construct_assistant_message(
            second_row["idea"]
        ),
    }
    for score_col in SCORE_COLUMNS:
        first_score, second_score = first_row[score_col], second_row[score_col]
        pair[f"{score_col}_{FIRST_IDEA_SUFFIX}"] = first_score
        pair[f"{score_col}_{SECOND_IDEA_SUFFIX}"] = second_score
        if first_score > second_score:
            pair[f"winner_{score_col}"] = FIRST_IDEA_SUFFIX
        elif first_score < second_score:
            pair[f"winner_{score_col}"] = SECOND_IDEA_SUFFIX
        else:
            pair[f"winner_{score_col}"] = "tie"
    return pair


def per_group_construction(group_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    # Construct group-level info dict (same for all pairs)
    first_row = group_df.iloc[0]
    group_id = first_row["id"]
    group_rater = first_row["rater"]
    group_user_message = construct_user_message(
        first_row["title"], first_row["context_puzzle"], cfg
    )

    if len(group_df) < 2:
        return pd.DataFrame()
    # pre-extract the arrays
    ideas = group_df["idea"].to_numpy()
    scores = {c: group_df[c].to_numpy() for c in SCORE_COLUMNS}

    first_idx, second_idx = np.triu_indices(len(group_df), k=1)
    m = len(first_idx)

    # construct the pairwise dataframe
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


def per_group_construction_adjacent(
    group_df: pd.DataFrame, cfg: DictConfig
) -> pd.DataFrame:
    """
    Sort the responses by their average human score (across novelty, probability, feasibility),
    then only create pairs between adjacent ranks: Rank 1 vs 2, Rank 2 vs 3, etc.
    """
    if len(group_df) < 2:
        return pd.DataFrame()

    # Compute average score across all score columns and sort descending
    group_df = group_df.copy()
    group_df["avg_score"] = group_df[SCORE_COLUMNS].mean(axis=1)
    group_df = group_df.sort_values("avg_score", ascending=False).reset_index(drop=True)

    first_row = group_df.iloc[0]
    group_id = first_row["id"]
    group_rater = first_row["rater"]
    group_user_message = construct_user_message(
        first_row["title"], first_row["context_puzzle"], cfg
    )

    ideas = group_df["idea"].to_numpy()
    scores = {c: group_df[c].to_numpy() for c in SCORE_COLUMNS}
    m = len(group_df) - 1  # number of adjacent pairs

    # Adjacent pairs: (0,1), (1,2), (2,3), (3,4)
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


def per_group_construction_extreme(
    group_df: pd.DataFrame, cfg: DictConfig
) -> pd.DataFrame:
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
        best_row["title"], best_row["context_puzzle"], cfg
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
    raw_df: pd.DataFrame, cfg, group_col: str = "id"
) -> pd.DataFrame:
    strategy = cfg.get("pairing_strategy", "exhaustive")
    if strategy == "adjacent":
        constructor = per_group_construction_adjacent
    elif strategy == "extreme":
        constructor = per_group_construction_extreme
    else:
        constructor = per_group_construction

    parts = []
    for _, g in tqdm(
        raw_df.groupby(group_col, sort=False), desc=f"Constructing dataset ({strategy})"
    ):
        df_g = constructor(g, cfg)
        if not df_g.empty:
            parts.append(df_g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@hydra.main(
    version_base=None, config_path="../config", config_name="dataset_building.yaml"
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # fetch the raw daata
    raw_df = load_raw_data(cfg)
    # perform the train test split (using the cfg.split settings)
    group_ids = pd.Series(raw_df["id"].unique())
    logger.info(f"Loaded {len(raw_df)} ideas from {len(group_ids)} groups")

    train_ids, forced_test_ids, extra_test_ids = category_aware_train_test_split(
        group_ids, **cfg.split
    )
    logger.info(
        f"Split into {len(train_ids)} train groups, {len(forced_test_ids)} forced test groups, {len(extra_test_ids)} extra test groups"
    )

    train_df = raw_df[raw_df["id"].isin(train_ids)]
    forced_test_df = raw_df[raw_df["id"].isin(forced_test_ids)]
    extra_test_df = raw_df[raw_df["id"].isin(extra_test_ids)]

    # construct the dataset
    train_built_df = dataset_construction(train_df, cfg)
    forced_test_built_df = dataset_construction(forced_test_df, cfg)
    extra_test_built_df = dataset_construction(extra_test_df, cfg)
    logger.info(
        f"Built {len(train_built_df)} train rows, {len(forced_test_built_df)} forced test rows, {len(extra_test_built_df)} extra test rows"
    )

    # save the dataset
    train_path = output_path_helper("built_train.csv", cfg)
    logger.info(f"Saving train dataset to {train_path}")
    train_built_df.to_csv(train_path, index=False)

    if cfg.output.combine_test_sets:
        test_built_df = pd.concat(
            [forced_test_built_df, extra_test_built_df], ignore_index=True
        )
        logger.info(f"Combined test datasets into {len(test_built_df)} rows")

        test_path = output_path_helper("built_test.csv", cfg)
        logger.info(f"Saving test dataset to {test_path}")
        test_built_df.to_csv(test_path, index=False)
    else:
        test_forced_path: str = output_path_helper("built_forced_test.csv", cfg)
        test_extra_path: str = output_path_helper("built_extra_test.csv", cfg)
        logger.info(f"Saving forced test dataset to {test_forced_path}")
        logger.info(f"Saving extra test dataset to {test_extra_path}")
        forced_test_built_df.to_csv(test_forced_path, index=False)
        extra_test_built_df.to_csv(test_extra_path, index=False)

    # optionally upload to Hugging Face Hub
    if cfg.huggingface.upload:
        split_paths = {"train": train_path}
        if cfg.output.combine_test_sets:
            split_paths["test"] = test_path
        else:
            split_paths["forced_test"] = test_forced_path
            split_paths["extra_test"] = test_extra_path
        upload_splits_to_hub(split_paths, cfg)


if __name__ == "__main__":
    main()
