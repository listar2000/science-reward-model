"""
Main training script for the multi-dimensional Science Reward Model.

Usage:
    python -m training.train                           # uses config/training.yaml defaults
    python -m training.train training.learning_rate=5e-5  # override via CLI
    python -m training.train dimensions.weights=[1.0,0.0,1.0]  # ignore feasibility
"""

import os

import hydra
import pandas as pd
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, set_seed

from trl import RewardConfig
from training.science_reward_trainer import ScienceRewardTrainer


def load_csv_dataset(path: str) -> Dataset:
    """Load a CSV file into a HuggingFace Dataset, keeping all columns as strings
    (the trainer's tokenize_and_extract will parse them)."""
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)


def build_reward_config(cfg: DictConfig) -> RewardConfig:
    """Translate the Hydra config's `training` section into a RewardConfig."""
    training_cfg: dict = OmegaConf.to_container(cfg.training, resolve=True)
    reward_config = RewardConfig(**training_cfg)
    return reward_config


def build_lora_config(cfg: DictConfig) -> LoraConfig | None:
    """Build a PEFT LoraConfig from the Hydra config's `lora` section.

    Returns None when lora_rank is 0 (full parameter training).
    """
    lora_rank: int = cfg.lora.lora_rank
    if lora_rank == 0:
        return None

    target_modules = list(cfg.lora.target_modules)
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=target_modules,
    )


def build_model(cfg: DictConfig, seed: int = 42) -> AutoModelForSequenceClassification:
    """Pre-instantiate the model with the correct num_labels.

    The upstream RewardTrainer hardcodes num_labels=1 in from_pretrained, so we
    create the model ourselves and pass the object (not a string) to the trainer,
    which makes the trainer skip its own from_pretrained call.
    """
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float32,
    )
    return model


@hydra.main(config_path="../config", config_name="training", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ---- Environment Variables ----
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    # ---- Config ----
    reward_config = build_reward_config(cfg)

    # ---- Data ----
    train_dataset = load_csv_dataset(cfg.data.train_file)
    eval_dataset = None
    if cfg.data.eval_file is not None:
        eval_file_cfg = OmegaConf.to_container(cfg.data.eval_file, resolve=True)
        if isinstance(eval_file_cfg, str):
            # Single eval dataset
            eval_dataset = load_csv_dataset(eval_file_cfg)
        elif isinstance(eval_file_cfg, dict):
            # Multiple eval datasets: {name: path, ...}
            eval_dataset = {
                name: load_csv_dataset(path) for name, path in eval_file_cfg.items()
            }
        else:
            raise ValueError(
                f"data.eval_file must be a string path or a mapping of name → path, "
                f"got {type(eval_file_cfg)}"
            )

    # ---- Dimension settings ----
    dim_names = list(cfg.dimensions.names)
    dim_weights = list(cfg.dimensions.weights)
    use_margins = cfg.dimensions.use_margins
    margin_weight = float(cfg.dimensions.get("margin_weight", 1.0))

    # ---- LoRA (optional) ----
    peft_config = build_lora_config(cfg)

    # ---- Model (pre-instantiate to control num_labels) ----
    model = build_model(cfg, seed=reward_config.seed)

    # ---- Trainer ----
    trainer = ScienceRewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dim_names=dim_names,
        dim_weights=dim_weights,
        use_margins=use_margins,
        margin_weight=margin_weight,
    )

    # ---- Train ----
    trainer.train()

    trainer.accelerator.print("✅ Training completed.")

    # ---- Save ----
    if cfg.save_model:
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
