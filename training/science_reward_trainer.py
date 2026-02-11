"""
ScienceRewardTrainer: a subclass of TRL's RewardTrainer for multi-dimensional
Bradley-Terry reward modelling.

Key differences from the default RewardTrainer:
  1. The model outputs a score vector of size `num_labels` (default 3) instead of a scalar.
  2. The dataset has "left" and "right" responses -- not "chosen" / "rejected" -- because
     a response can win on one dimension but lose on another.
  3. The data collator stacks left/right (instead of chosen/rejected) and carries
     per-dimension signs and margins.
  4. The loss is a *weighted* combination of per-dimension Bradley-Terry losses, where
     dimension weights are a configurable hyperparameter.
  5. Margins are 3-d vectors (absolute score differences) instead of scalars.
"""

import ast

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset

from trl import RewardTrainer, RewardConfig
from typing import Mapping

from training.data_collator import DataCollatorForMultiDimPreference


logger = get_logger(__name__)

# Default dimension names matching our CSV schema
DEFAULT_DIMENSIONS = ["novelty", "feasibility", "probability"]


def get_dataset_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    return (
        list(next(iter(dataset)).keys())
        if dataset.column_names is None
        else dataset.column_names
    )


def remove_none_values(example):
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).
    """
    if isinstance(example, list):
        return [
            remove_none_values(value) if isinstance(value, (dict, list)) else value
            for value in example
        ]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


class ScienceRewardTrainer(RewardTrainer):
    """
    Trainer for multi-dimensional Bradley-Terry reward models.

    In addition to the standard RewardTrainer arguments, accepts:
      - dim_names:    list of dimension names (default: novelty, feasibility, probability)
      - dim_weights:  list of floats weighting each dimension in the final loss
      - use_margins:  whether to use absolute score differences as BT margins
    """

    def __init__(
        self,
        dim_names: list[str] | None = None,
        dim_weights: list[float] | None = None,
        use_margins: bool = True,
        **kwargs,
    ):
        # Store multi-dim config *before* super().__init__ because it calls _prepare_dataset
        self.dim_names = dim_names or DEFAULT_DIMENSIONS
        self.dim_weights_list = dim_weights or [1.0] * len(self.dim_names)
        self.use_margins = use_margins

        if len(self.dim_weights_list) != len(self.dim_names):
            raise ValueError(
                f"dim_weights length ({len(self.dim_weights_list)}) must match "
                f"dim_names length ({len(self.dim_names)})"
            )

        # Let the parent handle model / tokenizer / PEFT / etc.
        # The parent will also create a DataCollatorForPreference -- we replace it below.
        super().__init__(**kwargs)

        # Replace the data collator with our multi-dimensional version
        self.data_collator = DataCollatorForMultiDimPreference(
            pad_token_id=self.processing_class.pad_token_id,
            pad_to_multiple_of=self.args.pad_to_multiple_of,
        )

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class,
        args: RewardConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Tokenise left/right responses and compute per-dimension signs & margins."""

        # Strip None values (same as parent)
        if isinstance(dataset, Dataset):
            dataset = dataset.with_transform(remove_none_values)

        column_names = get_dataset_column_names(dataset)

        # If already preprocessed, return as-is
        if "left_input_ids" in column_names and "right_input_ids" in column_names:
            return dataset

        # ---- Tokenisation + sign/margin computation ----

        map_kwargs: dict = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

        dim_names = self.dim_names
        use_margins = self.use_margins

        def tokenize_and_extract(example, processing_class):
            # --- Parse stringified conversation lists from CSV ---
            user_msg = ast.literal_eval(example["user_message"])
            left_msg = ast.literal_eval(example["assistant_message_left"])
            right_msg = ast.literal_eval(example["assistant_message_right"])

            left_conversation = user_msg + left_msg
            right_conversation = user_msg + right_msg

            # Tokenise via chat template
            left_ids = processing_class.apply_chat_template(
                left_conversation, return_dict=True
            )["input_ids"]
            right_ids = processing_class.apply_chat_template(
                right_conversation, return_dict=True
            )["input_ids"]

            # --- Per-dimension signs and margins ---
            signs: list[float] = []
            margins: list[float] = []

            for dim in dim_names:
                winner = example[f"winner_{dim}"]
                score_left = float(example[f"{dim}_left"])
                score_right = float(example[f"{dim}_right"])

                if winner == "left":
                    signs.append(1.0)
                elif winner == "right":
                    signs.append(-1.0)
                else:  # "tie"
                    signs.append(0.0)

                margins.append(abs(score_left - score_right))

            output = {
                "left_input_ids": left_ids,
                "right_input_ids": right_ids,
                "signs": signs,
            }
            if use_margins:
                output["margins"] = margins

            return output

        dataset = dataset.map(
            tokenize_and_extract,
            fn_kwargs={"processing_class": processing_class},
            **map_kwargs,
        )

        # ---- Filter by max_length ----
        if args.max_length is not None:
            filter_kwargs: dict = {}
            if isinstance(dataset, Dataset):
                filter_kwargs["num_proc"] = args.dataset_num_proc
                filter_kwargs["desc"] = (
                    f"Filtering {dataset_name} >{args.max_length} tokens"
                )
            dataset = dataset.filter(
                lambda ex: (
                    len(ex["left_input_ids"]) <= args.max_length
                    and len(ex["right_input_ids"]) <= args.max_length
                ),
                **filter_kwargs,
            )

        return dataset

    # ------------------------------------------------------------------
    # Signature columns (tells Trainer which columns to keep)
    # ------------------------------------------------------------------

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "left_input_ids",
                "right_input_ids",
                "signs",
                "margins",
            ]

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if self.model.training else "eval"

        inputs["use_cache"] = False
        outputs = model(**inputs)

        # outputs.logits shape: [2 * batch_size, num_dims]
        # First half = left responses, second half = right responses
        rewards_left, rewards_right = torch.chunk(outputs.logits, chunks=2)
        # Each: [batch_size, num_dims]

        signs = inputs["signs"]  # [batch_size, num_dims]

        # signed_diff: positive when the winner's reward is higher
        diff = rewards_left - rewards_right  # [batch_size, num_dims]
        signed_diff = signs * diff  # [batch_size, num_dims]

        # BT loss per sample per dimension
        if "margins" in inputs:
            per_sample_loss = -nn.functional.logsigmoid(signed_diff - inputs["margins"])
        else:
            per_sample_loss = -nn.functional.logsigmoid(signed_diff)

        # Mask out ties (sign == 0): no supervision signal for that dimension
        non_tie_mask = (signs != 0).float()  # [batch_size, num_dims]
        per_sample_loss = per_sample_loss * non_tie_mask

        # Per-dimension average loss (avoid division by zero)
        per_dim_loss = per_sample_loss.sum(dim=0) / non_tie_mask.sum(dim=0).clamp(min=1)

        # Weighted combination across dimensions
        dim_weights = torch.tensor(
            self.dim_weights_list, device=per_dim_loss.device, dtype=per_dim_loss.dtype
        )
        loss = (per_dim_loss * dim_weights).sum()

        # Optional reward centering (inherited from parent config)
        if self.args.center_rewards_coefficient is not None:
            all_rewards = torch.cat([rewards_left, rewards_right], dim=0)
            loss += self.args.center_rewards_coefficient * torch.mean(all_rewards**2)

        # ---- Metrics ----
        if mode == "train":
            num_tokens = (
                self.accelerator.gather_for_metrics(inputs["attention_mask"].sum())
                .sum()
                .item()
            )
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        with torch.no_grad():
            all_rewards = self.accelerator.gather(outputs.logits)
            self._metrics[mode]["min_reward"].append(all_rewards.min().item())
            self._metrics[mode]["mean_reward"].append(all_rewards.mean().item())
            self._metrics[mode]["max_reward"].append(all_rewards.max().item())

            # Per-dimension accuracy (fraction of non-tie pairs where winner scored higher)
            correct = (signed_diff > 0).float() * non_tie_mask
            for i, dim in enumerate(self.dim_names):
                dim_correct = correct[:, i].sum()
                dim_total = non_tie_mask[:, i].sum().clamp(min=1)
                acc = (
                    self.accelerator.gather_for_metrics(dim_correct / dim_total)
                    .mean()
                    .item()
                )
                self._metrics[mode][f"accuracy_{dim}"].append(acc)

                dim_margin = (signed_diff[:, i] * non_tie_mask[:, i]).sum() / dim_total
                dim_margin = (
                    self.accelerator.gather_for_metrics(dim_margin).mean().item()
                )
                self._metrics[mode][f"margin_{dim}"].append(dim_margin)

            # Overall accuracy (across all non-tie dimension-pairs)
            overall_acc = correct.sum() / non_tie_mask.sum().clamp(min=1)
            overall_acc = self.accelerator.gather_for_metrics(overall_acc).mean().item()
            self._metrics[mode]["accuracy"].append(overall_acc)

        return (loss, outputs) if return_outputs else loss
