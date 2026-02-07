import pandas as pd
import re


CATEGORY_MAP = {
    "biology_": "bio",
    "klab_": "soc",
    "css_": "soc",
    "ed": "soc",
    "phili": "soc",
    "psy": "soc",
    "chem": "natural",
    "earth": "natural",
    "energy": "natural",
}


def _extract_category(group_id: str) -> str:
    """
    Extract the category from the group id.
    """
    match = re.search(r".*?\d", group_id)
    if match:
        raw_category = match.group(0)[:-1]
        return CATEGORY_MAP.get(raw_category, raw_category)
    else:
        return "unknown"


def count_ideas_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of idea groups per category.

    Steps:
        1. Group rows by the ``id`` column.
        2. Extract the *category* from each group's ``id`` — the substring
           before the first digit (e.g. ``"bio10001"`` -> ``"bio"``).
        3. Count groups per category.

    Args:
        df: DataFrame that contains an ``id`` column with group identifiers.

    Returns:
        A two-column DataFrame ``["category", "count"]`` sorted in ascending
        order of ``count``.
    """
    # Deduplicate to one row per group, then extract categories
    unique_ids = df.groupby("id", sort=False).first().index.to_series()
    categories = unique_ids.apply(_extract_category)

    # Count and sort
    counts = (
        categories.value_counts()
        .reset_index()
        .rename(
            columns={
                "index": "category",
                _extract_category.__name__: "category",
                "count": "count",
            }
        )
    )
    counts.columns = ["category", "count"]
    return counts.sort_values("count").reset_index(drop=True)


def category_aware_train_test_split(
    group_ids: pd.Series,
    test_ratio: float = 0.1,
    test_categories: list[str] | None = None,
    even_split_per_category: bool = False,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Perform a category-aware train/test split on group IDs.

    If ``test_categories`` is provided, every group belonging to those
    categories is placed exclusively in the test set.  The remaining
    test-set budget (``target = len(group_ids) * test_ratio`` minus the
    forced groups) is then filled from the other categories.

    Args:
        group_ids: Series of group-ID strings (one entry per group).
        test_ratio: Desired fraction of *all* groups in the test set.
        test_categories: Categories whose groups are forced into the test
            set.  Pass ``None`` (default) to skip forced assignment.
        even_split_per_category: If ``True``, sample from each remaining
            category proportionally (using the budget-adjusted ratio).
            If ``False``, pool all remaining groups and sample uniformly.
        random_state: Seed for reproducible sampling.

    Returns:
        A tuple ``(train_ids, forced_test_ids, extra_test_ids)`` where each element is a Series
        of group-ID strings.
    """
    group_categories = group_ids.apply(_extract_category)

    total = len(group_ids)
    target_test_count = int(total * test_ratio)

    if test_categories is None:
        test_categories = []

    # Step 1 – force all groups in the designated categories into the test set
    forced_mask = group_categories.isin(test_categories)
    forced_test_idx = group_ids.index[forced_mask]
    remaining_idx = group_ids.index[~forced_mask]

    # Step 2 – determine remaining budget
    remaining_budget = max(0, target_test_count - len(forced_test_idx))

    # Step 3 – fill the remaining budget from other categories
    extra_test_idx: pd.Index = pd.Index([])

    if remaining_budget > 0 and len(remaining_idx) > 0:
        remaining_ids = group_ids.loc[remaining_idx]
        remaining_cats = group_categories.loc[remaining_idx]

        if even_split_per_category:
            effective_ratio = remaining_budget / len(remaining_ids)
            sampled_parts: list[pd.Index] = []
            for cat in sorted(remaining_cats.unique()):
                cat_idx = remaining_ids.index[remaining_cats == cat]
                n_sample = max(1, round(len(cat_idx) * effective_ratio))
                n_sample = min(n_sample, len(cat_idx))
                sampled = (
                    cat_idx.to_series()
                    .sample(n=n_sample, random_state=random_state)
                    .index
                )
                sampled_parts.append(sampled)
            extra_test_idx = (
                sampled_parts[0].append(sampled_parts[1:])
                if sampled_parts
                else pd.Index([])
            )
        else:
            n_sample = min(remaining_budget, len(remaining_ids))
            extra_test_idx = remaining_ids.sample(
                n=n_sample, random_state=random_state
            ).index

    # Combine and split
    all_test_idx = forced_test_idx.append(extra_test_idx)
    train_ids = group_ids.drop(all_test_idx).reset_index(drop=True)
    forced_test_ids = group_ids.loc[forced_test_idx].reset_index(drop=True)
    extra_test_ids = group_ids.loc[extra_test_idx].reset_index(drop=True)

    return train_ids, forced_test_ids, extra_test_ids


def main():
    df = pd.read_csv("data/train.csv")
    print(count_ideas_by_category(df))


if __name__ == "__main__":
    main()
