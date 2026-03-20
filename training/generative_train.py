"""
Main training script for supervised fine-tuning of a generative model.

This script targets datasets produced by
`dataset_building/build_dataset_generative_training.py`, which emit one
`messages` column per training example.

Usage:
    python -m training.generative_train
    python -m training.generative_train training.learning_rate=5e-5
    python -m training.generative_train data.train_file=data/processed_generative_biology/built_train.csv
"""

import ast
import os
from collections.abc import Mapping

import hydra
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer


def load_csv_dataset(path: str) -> Dataset:
    """Load a CSV file into a HuggingFace Dataset and parse the `messages` column."""
    df = pd.read_csv(path)
    df["messages"] = df["messages"].apply(ast.literal_eval)
    return Dataset.from_pandas(df)


def build_training_arguments(cfg: DictConfig) -> SFTConfig:
    """Translate the Hydra config's `training` section into an SFTConfig."""
    training_cfg: dict = OmegaConf.to_container(cfg.training, resolve=True)
    return SFTConfig(**training_cfg)


def build_lora_config(cfg: DictConfig) -> LoraConfig | None:
    """Build a PEFT LoraConfig from the Hydra config's `lora` section."""
    lora_rank: int = cfg.lora.lora_rank
    if lora_rank == 0:
        return None

    target_modules = list(cfg.lora.target_modules)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=target_modules,
    )




def resolve_eval_datasets(cfg: DictConfig) -> Mapping[str, str]:
    """Resolve evaluation file paths."""
    if cfg.data.eval_file is not None:
        eval_file_cfg = OmegaConf.to_container(cfg.data.eval_file, resolve=True)
        if isinstance(eval_file_cfg, str):
            return {"eval": eval_file_cfg}
        if isinstance(eval_file_cfg, dict):
            return dict(eval_file_cfg)
        raise ValueError(
            f"data.eval_file must be a string path or a mapping of name -> path, got {type(eval_file_cfg)}"
        )

    data_folder = os.path.dirname(cfg.data.train_file)
    discovered: dict[str, str] = {}
    for fname in sorted(os.listdir(data_folder)):
        if fname.startswith(("built_ID_test_", "built_OOD_test_")) and fname.endswith(
            ".csv"
        ):
            name = fname[len("built_") : -len(".csv")]
            discovered[name] = os.path.join(data_folder, fname)
    return discovered


@hydra.main(config_path="../config", config_name="generative_training", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    training_args = build_training_arguments(cfg)

    peft_config = build_lora_config(cfg)

    train_dataset = load_csv_dataset(cfg.data.train_file)

    eval_paths = resolve_eval_datasets(cfg)
    eval_dataset = None
    if eval_paths:
        eval_dataset = {
            name: load_csv_dataset(path) for name, path in eval_paths.items()
        }

    trainer = SFTTrainer(
        model=cfg.model.name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.accelerator.print("Training completed.")

    if cfg.save_model:
        trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"Model saved to {training_args.output_dir}")

    if cfg.training.push_to_hub:
        trainer.push_to_hub()
        trainer.accelerator.print("Model pushed to the Hugging Face Hub.")


if __name__ == "__main__":
    main()
