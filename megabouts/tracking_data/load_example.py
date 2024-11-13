import os
import pandas as pd


def load_example_data(source):
    """
    Load tracking data from various sources.

    Args:
        source (str): The source of the tracking data. Options are:
                      'fulltracking_posture', 'SLEAP_fulltracking', 'HR_DLC', 'zebrabox_SLEAP'.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The loaded tracking data.
            - fps (int): The frames per second of the data.
            - mm_per_unit (float): The millimeters per unit of the data.

    Raises:
        ValueError: If an invalid source is provided.

    Examples:
        >>> df, fps, mm_per_unit = load_example_data('fulltracking_posture')
        >>> df, fps, mm_per_unit = load_example_data('SLEAP_fulltracking')
    """
    sources = {
        "fulltracking_posture": {
            "file": "example_highres_posture_700fps.csv",
            "fps": 700,
            "mm_per_unit": 1,
        },
        "SLEAP_fulltracking": {
            "file": "example_fulltracking_SLEAP_350fps.csv",
            "fps": 350,
            "mm_per_unit": 0.025,
        },
        "HR_DLC": {
            "file": "example_headrestrained_DLC_250fps.csv",
            "fps": 250,
            "mm_per_unit": 0.01578,
        },
        "zebrabox_SLEAP": {
            "file": "example_zebrabox_SLEAP_25fps.csv",
            "fps": 25,
            "mm_per_unit": 0.11,
        },
    }

    if source not in sources:
        raise ValueError(
            "Invalid source provided. Available sources are: 'fulltracking_posture', 'SLEAP_fulltracking', 'HR_DLC', 'zebrabox_SLEAP'."
        )

    metadata = sources[source]
    csv_path = os.path.join(
        os.path.dirname(__file__), "example_dataset", metadata["file"]
    )
    if source == "HR_DLC":
        df = pd.read_csv(csv_path, header=[0, 1, 2])
    else:
        df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric)

    return df, metadata["fps"], metadata["mm_per_unit"]
