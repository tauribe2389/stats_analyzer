from __future__ import annotations

from typing import Any


class DataSplitter:
    """Split a dataset by the configured group variables."""

    def split(self, dataset: Any, group_variables: list[str]) -> dict[tuple[Any, ...], Any]:
        if not group_variables:
            return {("ALL",): dataset}

        split_data: dict[tuple[Any, ...], Any] = {}
        # Older pandas releases do not support the dropna keyword in groupby.
        try:
            grouped = dataset.groupby(group_variables, dropna=False)
        except TypeError:
            grouped = dataset.groupby(group_variables)

        for key, frame in grouped:
            tuple_key = key if isinstance(key, tuple) else (key,)
            split_data[tuple_key] = frame
        return split_data
