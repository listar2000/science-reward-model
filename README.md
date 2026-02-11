## Scientific Idea Reward Modelling

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

this ensures that your code will be auto-formatted and checked for errors before your commits take effect.

### File Structure

The repo is organized by the different stages of the reward modelling pipeline: `dataset_building`, reward model `training`, and `evaluation`. A overview of the file structure is as follows:

```
.
├── dataset_building/  # <- all the code for building the dataset from raw csv file
├── training/  # <- all the code for training the reward model
├── evaluation/  # <- all the code for evaluating the reward model
├── config/  # <- the yaml configuration files for all stages (see below)
├── data/  # <- the raw and processed dataset (not tracked by git)
├── requirements.txt  # <- the dependencies for the project
├── README.md  # <- this file
├── .gitignore  # <- the gitignore file
├── .pre-commit-config.yaml  # <- the pre-commit config file
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
    
    # fetch the raw daata
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