## Reward Model Training Infrastructure

This document outlines the changes needed to support a customized Bradley-Terry reward model training -- based on Huggingface TRL.

### Related Files

- Existing TRL reward trainer and data collator: `./trl/trl/trainer/reward_trainer.py`
- Existing TRL reward configuration: `./trl/trl/trainer/reward_config.py`
- Example of customized data file we will work with: `./data/processed/built_extra_test.csv` (used to understand column names, etc.)

### Overview

We want to train a reward model (i.e. a decoder-only transformer but with last layer outputting scalar scores, `AutoModelForSequenceClassification`) via Bradley-Terry-type training over pairwise responses. The default trainer for this is the `RewardTrainer` class in TRL. But our dataset has one particular feature:

**Each comparison (row) is (human)-rated on three, not just one, dimensions -- novelty, feasibility, and probability -- each is represented as an integer score**.

This important feature leads to the following technical challenges that make the default trainer infeasible:

1. The output head of the reward model should be of dimension 3, not 1.
2. The default trainer or data preprocessing logic assumes the existence of "chosen" and "rejected" columns in the original dataset (or after applying some transformations). But for our dataset, we do not have a "chosen" response -- since a response can win on one dimension but lose on another.
3. Similarly, the data collator (used after the preprocessing) will concatenate the token ids of the chosen response to the left and the rejected response to the right.
4. In the loss function, the loss is simply computed via the (single) difference between the scores of the winning ("chosen") and losing ("rejected") responses. But for our dataset, the final loss should be a weighted combination (weight as a hyperparameter) of the Bradley-Terry losses for all three dimensions.
5. Finally, the original trainer also allows a (scalar) "margin" column in the dataset for each row -- used in the loss computation. But for our case, the margin should also be a vector of three scalars, one for each dimension.

Below we give a slightly more detailed guidance on the points above (mainly where you should look into and explore yourself):

#### Point 1: output head dimension

This should be the most straightforward to deal with. The `AutoModelForSequenceClassification` class does allow a `num_labels` argument to be passed as a part of the model initialization arguments. But this argument is not in the `RewardConfig` class. So you should try to understand how to pass this in.

#### Point 2: no "chosen" response

Look at the `_prepare_dataset` method in the `RewardTrainer` class.

#### Point 3: data collator

Look at the `DataCollatorForPreference` class in the `RewardTrainer` class.

#### Point 4: loss function

Look at the `compute_loss` method in the `RewardTrainer` class.

#### Point 5: margin

Look at the `compute_loss` method and `DataCollatorForPreference` class in the `RewardTrainer` class.

---

### Implementation Notes

You can feel free to explore the files and decide what you will need to change. But the recommendation is that:

- Using the hydra to handle config and put hyperparameters in the `config/training.yaml` file.
- Have a `training/science_reward_trainer.py` file that creates a subclass of the `RewardTrainer`. Only override the methods that need customization, for instance, the `_prepare_dataset`, `compute_loss` (and others if needed).
- Have a customized data collator in the `training/data_collator.py` file. This should be sth similar to the `DataCollatorForPreference` class, but with the ability to handle the three dimensions.
- Have some high-level hyper-parameters (e.g. the weights for the three dimensions) in the `config/training.yaml` file.

You can assume that the original data file's columns are fixed (i.e. can be treated as constants). But the new training system should support cases where we only want to care about a subset of the dimensions (maybe achievable through setting weights to zero).

Also **make sure** you understand the existing training system and how it works -- beyond what I've outlined above. Raise questions if you are not sure about something. Finally, check out the reference data file listed above (take the first few lines as this file is large).