"""
Main training script for the multi-dimensional Science Reward Model.

Usage:
    python -m training.train                           # uses config/training.yaml defaults
    python -m training.train training.learning_rate=5e-5  # override via CLI
    python -m training.train dimensions.weights=[1.0,0.0,1.0]  # ignore feasibility
"""

import hydra
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from trl.trainer.reward_config import RewardConfig
from training.science_reward_trainer import ScienceRewardTrainer

import os


def load_csv_dataset(path: str) -> Dataset:
    """Load a CSV file into a HuggingFace Dataset, keeping all columns as strings
    (the trainer's tokenize_and_extract will parse them)."""
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)


def build_reward_config(cfg: DictConfig) -> RewardConfig:
    """Translate the Hydra config's `training` section into a RewardConfig."""
    training_cfg: dict = OmegaConf.to_container(cfg.training, resolve=True)  # type: ignore

    # Pop keys that aren't valid TrainingArguments fields
    # (we handle them separately or they map to model_init_kwargs)
    reward_config = RewardConfig(
        **training_cfg,
        # Pass num_labels so the classification head has the right size
        model_init_kwargs={"num_labels": cfg.model.num_labels},
    )
    return reward_config


@hydra.main(config_path="../config", config_name="training", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ---- Environment Variables ----
    os.environ.setdefault("WANDB_PROJECT", cfg.training.wandb_project)

    # ---- Config ----
    reward_config = build_reward_config(cfg)

    # ---- Data ----
    train_dataset = load_csv_dataset(cfg.data.train_file)
    eval_dataset = None
    if cfg.data.eval_file is not None:
        eval_dataset = load_csv_dataset(cfg.data.eval_file)

    # ---- Dimension settings ----
    dim_names = list(cfg.dimensions.names)
    dim_weights = list(cfg.dimensions.weights)
    use_margins = cfg.dimensions.use_margins

    # ---- Trainer ----
    trainer = ScienceRewardTrainer(
        model=cfg.model.name,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dim_names=dim_names,
        dim_weights=dim_weights,
        use_margins=use_margins,
    )

    # ---- Train ----
    trainer.train()

    trainer.accelerator.print("✅ Training completed.")

    # ---- Save ----
    trainer.save_model(reward_config.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {reward_config.output_dir}")

    # ---- Maybe Push to Hub ----
    if reward_config.push_to_hub:
        trainer.push_to_hub()
        trainer.accelerator.print(
            f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}."
        )


if __name__ == "__main__":
    main()
