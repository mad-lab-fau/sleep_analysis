import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocess_movement_df(radar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess radar data to extract aggregated movement in 30s epochs
    radar_df: radar data
    return: preprocessed radar data as DataFrame
    """

    # Moving average
    radar_df = _moving_average(radar_df, window_size=19000, channel="I")

    # Scale
    radar_df = _scale_df(radar_df, channel="MA")

    # first order derivative
    radar_df = _first_order_derivative(radar_df, channel="MA_scaled")

    # normalize
    radar_df = _normalize(radar_df, channel="FOD")

    # thresholding
    radar_df = _threshold(radar_df, channel="NORM")

    # group in 30s epochs
    radar_df = _group_30s(radar_df, channel="threshold")

    return radar_df[["grouped"]]


def preprocess_movement(radar_dict: dict) -> pd.DataFrame:
    """
    Combine radar data DataFrames stored in a dictonary to extract the combined aggregated movement in 30s epochs
    radar_dict: dictionary containing radar data DataFrames

    return: preprocessed radar data of combined radar datastreams
    """
    # Calculate movement for each radar datastream
    for key, radar_df in radar_dict.items():
        radar_dict[key] = preprocess_movement_df(radar_df)

    # combine movement radar datastreams
    radar_df = pd.concat(radar_dict.values(), axis=1)

    # take the max of the movement datastreams
    movement_df = radar_df.max(axis=1).to_frame(name="movement")

    return movement_df


def _moving_average(df, window_size, channel):
    """
    Apply a moving average filter to a specified channel in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing signal data.
    - window_size (int): Size of the moving average window.
    - channel (str): The key representing the channel in the DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing the original data and the moving average in the 'MA' column.

    Notes:
    - Applies a moving average filter to the specified channel in the DataFrame.
    - The first and last values are padded to handle edge effects.
    - The resulting moving average values are stored in a new column named 'MA' in the original DataFrame.
    """

    if channel not in df.columns:
        raise ValueError(f"Channel '{channel}' not found in the DataFrame.")

    first_avg = int(df[channel].iloc[: window_size // 2].mean())

    last_avg = int(df[channel].iloc[-window_size // 2 :].mean())

    pad_array = np.array(df[channel])

    padded = np.pad(
        array=pad_array, pad_width=(window_size // 2), mode="constant", constant_values=(first_avg, last_avg)
    )

    # Assign the 'MA' column
    df = df.assign(MA=padded[window_size // 2 : len(padded) - window_size // 2])

    # Calculate the rolling mean
    df["MA"] = df["MA"].rolling(window=window_size, center=True, min_periods=1).mean()

    return df


def _scale_df(df, channel):
    x = df[[channel]].values  # returns a numpy array
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, index=df.index, columns=["MA_scaled"])
    ma_df = pd.concat([df, x_scaled_df], axis=1)

    return ma_df


def _first_order_derivative(df, channel):
    """
    Calculate the first-order derivative of a specified channel in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing signal data.
    - channel (str): The key representing the channel for which the derivative is calculated.

    Returns:
    - pd.DataFrame: DataFrame containing the original data and the calculated first-order derivative.

    Notes:
    - Calculates the first-order derivative of the specified channel using the pandas differentiate_signal function.
    - The absolute values of the derivative are stored in a new column named 'FOD' in the original DataFrame.

    Raises:
    - ValueError: If the specified channel is not found in the DataFrame.
    """
    # Check if the specified channel exists in the DataFrame
    if channel not in df.columns:
        raise ValueError(f"Channel '{channel}' not found in the DataFrame.")

    signal = df[channel]

    derivative = signal.diff()

    df.loc[:, "FOD"] = derivative.abs()

    return df


def _normalize(df, channel):
    """adjusts the amplitude of the signal, so that the max is at 1 and the rest between 0 and 1"""
    min_val = df[channel].min()
    max_val = df[channel].max()
    df["NORM"] = (df[channel] - min_val) / (max_val - min_val)
    return df


def _threshold(df, channel, threshold=0.2):
    """
    Applies a threshold to a specific column ('key') in the DataFrame ('df').

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - key (str): Column in the DataFrame to be thresholded.
    - threshold (float): Threshold value.

    Returns:
    - pd.DataFrame: DataFrame with threshold applied.
    """
    df["threshold"] = df[channel].where(df[channel] > threshold, 0.0)

    return df


def _group_30s(df, channel):
    df["grouped"] = df[channel].copy()
    df.index = df.index.floor("30s")
    df = df.groupby(pd.Grouper(freq="30s", origin="epoch")).mean()

    return df
