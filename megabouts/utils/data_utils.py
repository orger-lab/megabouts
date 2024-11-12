import numpy as np
import pandas as pd
from typing import List, Tuple


def create_hierarchical_df(
    data_info: List[Tuple[str, str, np.ndarray]],
) -> pd.DataFrame:
    """
    Create a hierarchical DataFrame from multiple data arrays, handling both one-dimensional and
    multi-dimensional arrays by reshaping one-dimensional arrays to two-dimensional if needed.

    Parameters
    ----------
    data_info : list of tuples
        Each tuple contains the data array and its metadata.
        Metadata format: ('main_label', 'sub_label', 'data_array')

    Returns
    -------
    pd.DataFrame
        The hierarchical pandas DataFrame.

    Raises
    ------
    ValueError
        If data_info is empty or data is not a numpy array.

    Examples
    --------
    >>> data1 = np.random.rand(100, 10)
    >>> data2 = np.random.rand(100, 10)
    >>> data3 = np.random.rand(100, 10)
    >>> data4 = np.random.rand(100)
    >>> data5 = np.random.rand(100)
    >>> data_info = [
    ...     ('label1', 'sub1', data1),
    ...     ('label2', 'sub1', data2),
    ...     ('label3', 'sub1', data3),
    ...     ('label4', 'None', data4),
    ...     ('label5', 'None', data5)
    ... ]
    >>> df = create_hierarchical_df(data_info)
    >>> print(df.head())
    """
    if not data_info:
        raise ValueError(
            "data_info is empty. Please provide at least one data array with its metadata."
        )

    frames = []

    for main_label, sub_label, data_array in data_info:
        if not isinstance(data_array, np.ndarray):
            raise ValueError(f"Data must be a numpy array, but got {type(data_array)}")

        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        num_samples = data_array.shape[0]

        if sub_label == "None":
            columns = pd.MultiIndex.from_product([[main_label], [""], [""]])
        else:
            columns = pd.MultiIndex.from_product(
                [
                    [main_label],
                    [sub_label],
                    [str(i) for i in range(data_array.shape[1])],
                ]
            )

        frame = pd.DataFrame(data_array, index=range(num_samples), columns=columns)
        frames.append(frame)

    df = pd.concat(frames, axis=1)

    # Remove names from the MultiIndex levels
    df.columns.names = [None, None, None]

    return df
