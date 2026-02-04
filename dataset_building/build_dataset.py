from omegaconf import DictConfig
import hydra
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


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


def dataset_construction(
    raw_df: pd.DataFrame, cfg, group_col: str = "id"
) -> pd.DataFrame:
    parts = []
    for _, g in tqdm(
        raw_df.groupby(group_col, sort=False), desc="Constructing dataset"
    ):
        df_g = per_group_construction(g, cfg)
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
    # construct the dataset
    dataset = dataset_construction(raw_df, cfg)
    # save the dataset
    output_path = os.path.join(cfg.output.output_data_folder, "dataset.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path} with {len(dataset)} rows")


if __name__ == "__main__":
    main()
