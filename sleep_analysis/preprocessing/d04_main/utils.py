def load_and_sync_radar_data(subj):

    radar_data = subj.radar_data

    if radar_data is None:
        return None

    print("sync radar data ...", flush=True)
    synced_radar = subj.sync_radar(radar_data)
    print("synced radar data ...", flush=True)

    return synced_radar
