"""Utilities for uploading datasets to the Hugging Face Hub."""

import argparse
import glob
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import DictConfig

logger = logging.getLogger("hf_utils")

BUILT_CSV_PREFIX = "built_"


def _get_hf_token() -> str:
    """Load the Hugging Face API token from the ``.env`` file.

    Looks for a variable named ``HF_TOKEN`` in the ``.env`` file located at
    the project root (one level above ``dataset_building/``).

    Returns:
        The token string.

    Raises:
        ValueError: If ``HF_TOKEN`` is not set or is empty.
    """
    # .env lives at project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            f"HF_TOKEN not found. Please set it in {env_path}. "
            "See .env_example for reference."
        )
    return token


def discover_splits(folder: str) -> dict[str, str]:
    """Discover dataset splits from ``built_*.csv`` files in *folder*.

    Each file matching ``built_<split_name>.csv`` is mapped to a split whose
    name is ``<split_name>`` (e.g. ``built_train.csv`` -> ``"train"``,
    ``built_forced_test.csv`` -> ``"forced_test"``).

    Args:
        folder: Path to the directory containing built CSV files.

    Returns:
        A dict mapping split names to their absolute file paths.

    Raises:
        FileNotFoundError: If *folder* does not exist or contains no matches.
    """
    pattern = os.path.join(folder, f"{BUILT_CSV_PREFIX}*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No files matching '{BUILT_CSV_PREFIX}*.csv' found in {folder}"
        )

    split_paths: dict[str, str] = {}
    for p in paths:
        filename = os.path.basename(p)  # e.g. "built_train.csv"
        split_name = filename[len(BUILT_CSV_PREFIX) : -len(".csv")]
        split_paths[split_name] = os.path.abspath(p)

    return split_paths


def _upload_splits(
    split_paths: dict[str, str],
    repo_id: str,
    private: bool = True,
) -> None:
    """Core upload logic shared by the config-driven and standalone paths.

    Args:
        split_paths: Mapping of ``split_name -> local_csv_path``.
        repo_id: Hugging Face dataset repository id (e.g. ``"user/dataset"``).
        private: Whether the repository should be private.
    """
    token = _get_hf_token()
    api = HfApi(token=token)

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info(f"Using HF dataset repo: {repo_id} (private={private})")

    for split_name, local_path in split_paths.items():
        if not os.path.isfile(local_path):
            logger.warning(f"Skipping '{split_name}': file not found at {local_path}")
            continue

        path_in_repo = f"{split_name}.csv"
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info(f"Uploaded {local_path} -> {repo_id}/{path_in_repo}")

    logger.info("All splits uploaded successfully.")


def upload_splits_to_hub(
    split_paths: dict[str, str],
    cfg: DictConfig,
) -> None:
    """Upload built CSV splits using Hydra config (called from ``build_dataset.py``).

    Thin wrapper around :func:`_upload_splits` that reads ``repo_id`` and
    ``private`` from the Hydra config.

    Args:
        split_paths: Mapping of ``split_name -> local_csv_path``.
        cfg: The Hydra config; expects ``cfg.huggingface.repo_id`` and
            ``cfg.huggingface.private``.
    """
    _upload_splits(
        split_paths,
        repo_id=cfg.huggingface.repo_id,
        private=cfg.huggingface.private,
    )


def main() -> None:
    """Standalone entry-point for uploading a folder of built CSVs to HF Hub.

    Usage::

        python hf_utils.py --folder data/processed --repo-id user/my-dataset [--private] [--public]
    """
    parser = argparse.ArgumentParser(
        description="Upload built_*.csv splits from a folder to Hugging Face Hub."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing built_*.csv files.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help='HF dataset repository id, e.g. "username/dataset-name".',
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make the HF repo public (default: private).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )

    split_paths = discover_splits(args.folder)
    logger.info(f"Discovered splits: {list(split_paths.keys())}")

    _upload_splits(split_paths, repo_id=args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
