# Contemporary AI Lacks the Imagination to Diverge or Negate in Science

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/) <!-- TODO: replace with the actual HF collection/model link -->

This is the official codebase for the manuscript **"Contemporary AI lacks the imagination to diverge or negate in science."** The project studies how human scientists evaluate research ideas along three dimensions -- **novelty**, **feasibility**, and **probability of success** -- and trains reward models that learn these human preferences from large-scale pairwise comparisons of scientific ideas.

The repository covers the full pipeline used in the paper:

- **Dataset building** -- turning raw human-rated comparisons into a pairwise preference dataset (`dataset_building/`).
- **Reward model training** -- a multi-dimensional Bradley-Terry reward model trained on those preferences (`training/`).
- **Analysis** -- a notebook reproducing the statistical analyses in the manuscript (`analysis/`).

### Model Checkpoints

The trained reward models are hosted on the Hugging Face Hub:

> 🤗 **[Hugging Face checkpoints](https://huggingface.co/)** &nbsp; https://huggingface.co/UchiKlab/science-reward-model &nbsp;*(link to be released)*

### Getting Started

#### Environment Setup

We recommend using `conda` to create a new environment with `Python 3.12`.

```bash
conda create -n science python=3.12
conda activate science
```

Then, install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

#### Precommit Setup

For better code quality, please install the pre-commit hooks by running:

```bash
pre-commit install
```

This ensures that your code will be auto-formatted and checked for errors before your commits take effect.

### File Structure

The repo is organized by the different stages of the reward modelling pipeline: `dataset_building`, reward model `training`, and `evaluation`. An overview of the file structure is as follows:

```
.
├── dataset_building/        # Code for constructing the dataset from raw CSV files
├── training/                # Reward model training pipeline
├── evaluation/              # Evaluation scripts and metrics
├── analysis/                # Notebooks for statistical analysis
├── config/                  # YAML configuration files for all stages (see below)
├── data/                    # Raw and processed datasets (not tracked by git)
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
└── .pre-commit-config.yaml  # Pre-commit configuration
└── .env_example             # Example environment variables file
```

### Configuration System

We use `hydra` as the configuration system, which uses `yaml`-based configuration files. Check out the [docs](https://hydra.cc/docs/intro/) for more details.

In short, the config for each component is stored within a `.yaml` file in the `config` folder. For example, the config for the `dataset_building` component is stored in `config/dataset_building.yaml`. An example of interfacing with the config is in `dataset_building/build_dataset.py`.

```python
@hydra.main(
    version_base=None, config_path="../config", config_name="dataset_building.yaml"
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # this helps resolve variables like ${x.y.z} to the actual values
    
    # fetch the raw data
    raw_df = load_raw_data(cfg)
    # ...rest of the code
```

The `hydra` system allows easy *overriding* of the config values. For instance, we can change the output data folder by running:

```bash
python dataset_building/build_dataset.py output.output_data_folder=data/processed_v2
```

This will override the `output_data_folder` in the `dataset_building.yaml` file to `data/processed_v2`.


### Building the Dataset

#### 1. Run the pipeline

With `train.csv` in place, build the pairwise dataset and perform the train/test split:

```bash
python dataset_building/build_dataset.py
```

Settings (split ratios, test categories, output paths, etc.) live in `config/dataset_building.yaml`.
Built CSVs are written to `data/processed/` as `built_train.csv`, `built_forced_test.csv`, and `built_extra_test.csv` by default (can also combine the two test sets via overriding the `output.combine_test_sets` setting).

#### 2. Upload to Hugging Face Hub

**Via the pipeline** — set `huggingface.upload: true` and fill in `huggingface.repo_id` in the config, then re-run the build command above.

**Standalone** — upload an existing folder of built CSVs directly:

```bash
python dataset_building/hf_utils.py --folder data/processed --repo-id user/dataset-name [--public]
```

Split names are inferred from filenames (`built_<name>.csv` → `<name>`).

Both methods read `HF_TOKEN` from the `.env` file at the project root (see `.env_example`).

### Training the Reward Model

We train a Bradley-Terry reward model on pairwise comparison data, where each pair is rated on **three dimensions**: novelty, feasibility, and probability. This differs from standard reward model training (which assumes a single chosen/rejected pair) in several ways:

- **Multi-dimensional output**: The model head outputs a 3-d score vector (one per dimension) instead of a scalar.
- **No fixed chosen/rejected ordering**: Each response can win on some dimensions and lose on others, so the trainer works with symmetric left/right pairs and per-dimension winner labels.
- **Weighted per-dimension loss**: The final loss is a weighted combination of per-dimension Bradley-Terry losses. Dimension weights are configurable -- setting a weight to zero lets you ignore a dimension entirely.
- **Vector margins**: Optionally uses the absolute human rating difference per dimension as a margin term in the BT loss.

#### Quick start

All settings live in `config/training.yaml`. To launch training with the defaults:

```bash
python -m training.train
```

Override any setting via the command line (Hydra syntax):

```bash
# Only train on novelty
python -m training.train dimensions.weights=[1.0,0.0,0.0]

# Change learning rate
python -m training.train training.learning_rate=5e-5

# Disable margin-augmented loss
python -m training.train dimensions.use_margins=false
```

When working with multiple GPUs, you can use the `accelerate` config file to configure the training. For example, to use 8 GPUs, you can run:

```bash
accelerate launch --config_file config/accelerate/fsdp2.yaml -m training.train
```

This will use the `fsdp2` config file, which is a pre-configured config file for training on 8 GPUs.

#### Key files

| File | Purpose |
|---|---|
| `config/training.yaml` | All hyperparameters (model, data paths, dimension weights, training args) |
| `training/train.py` | Hydra entry point -- loads data, builds config, runs the trainer |
| `training/science_reward_trainer.py` | `ScienceRewardTrainer`, a subclass of TRL's `RewardTrainer` with custom dataset preprocessing, multi-dim loss, and per-dimension metrics |
| `training/data_collator.py` | `DataCollatorForMultiDimPreference`, handles batching/padding of left-right pairs with 3-d signs and margins |

### Analysis

The `analysis/` folder contains the notebook used to produce the statistical
results in the manuscript:

| Notebook | Purpose |
|---|---|
| `analysis/regression_analysis.ipynb` | Regression models relating researcher seniority (citations, academic age, publication count) to idea-selection behavior and to rated novelty/feasibility/probability, as well as their interaction effects. Replication of SOTA models. |

The notebook reads the `train.csv` described in [`data/README.md`](data/README.md). See [`analysis/README.md`](analysis/README.md) for details.
