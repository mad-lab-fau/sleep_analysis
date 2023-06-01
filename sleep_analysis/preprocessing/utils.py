"""Helper functions for preprocessing."""
import mesa_data_importer as mesa
import pandas as pd


def extract_edf_channel(path, subj_id, channel, tmin=None, tmax=None):
    """Extract data from one channel of an EDF file."""
    if tmin and tmax:
        edf = mesa.load_edf(path, subj_id).crop(tmin=tmin, tmax=tmax)
    else:
        edf = mesa.load_edf(path, subj_id)

    channel_data = edf.pick_channels([channel])
    data = channel_data.get_data()[0, :]

    time, epochs = _create_datetime_index(channel_data.info["meas_date"], times_array=channel_data.times)
    if channel == "EKG":
        data = pd.DataFrame(data, index=time).rename(columns={0: "ecg"})
    else:
        data = pd.DataFrame(data, index=time).rename(columns={0: "resp"})
    return data, epochs


def _create_datetime_index(starttime, times_array):
    starttime_s = starttime.timestamp()  # * 1000000000
    times_array = times_array + starttime_s
    datetime_index = pd.to_datetime(times_array, unit="s")
    epochs = _generate_epochs(datetime_index)
    return datetime_index, epochs


def _generate_epochs(datetime_index):
    start_time = datetime_index[0]
    epochs_30s = datetime_index.round("30s")

    epochs_clear = (epochs_30s - start_time).total_seconds()
    # epochs_clear.total_seconds()

    epochs_clear = epochs_clear / 30
    epochs = epochs_clear.astype(int)
    return epochs
