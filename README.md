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
├── reward_model_training/  # <- all the code for training the reward model
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