"""
Custom data collator for multi-dimensional preference data.

Unlike the default DataCollatorForPreference (which assumes chosen/rejected pairs with a scalar
margin), this collator handles:
  - Left/right response pairs (no inherent chosen/rejected ordering)
  - Per-dimension signs indicating which response won (+1=left, -1=right, 0=tie)
  - Per-dimension margins (absolute score differences)
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers.data.data_collator import DataCollatorMixin

from trl.trainer.utils import pad


@dataclass
class DataCollatorForMultiDimPreference(DataCollatorMixin):
    """
    Data collator for multi-dimensional preference data.

    Each example should contain:
      - "left_input_ids":  token IDs for the left response
      - "right_input_ids": token IDs for the right response
      - "signs":   list of floats (+1, -1, 0) per dimension indicating the winner
      - "margins": list of floats (absolute score differences) per dimension (optional)

    The collator concatenates left and right input IDs (left first, right second) and pads
    to the maximum sequence length in the batch -- mirroring how the original TRL collator
    stacks chosen/rejected.

    Args:
        pad_token_id: Token ID used for padding.
        pad_to_multiple_of: If set, sequences are padded to a multiple of this value.
        return_tensors: Tensor type to return (only "pt" supported).
    """

    pad_token_id: int
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Collect token ID lists
        left_input_ids = [torch.tensor(ex["left_input_ids"]) for ex in examples]
        right_input_ids = [torch.tensor(ex["right_input_ids"]) for ex in examples]

        # Stack left then right (same layout as chosen + rejected in the original collator)
        input_ids = left_input_ids + right_input_ids
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        output: dict[str, Any] = {}

        # Pad sequences
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Per-dimension winner signs: [batch_size, num_dims]
        if "signs" in examples[0]:
            output["signs"] = torch.tensor(
                [ex["signs"] for ex in examples], dtype=torch.float
            )

        # Per-dimension margins (absolute score differences): [batch_size, num_dims]
        if "margins" in examples[0]:
            output["margins"] = torch.tensor(
                [ex["margins"] for ex in examples], dtype=torch.float
            )

        return output
