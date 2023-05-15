def process_actigraphy(df_actigraph, df_psg, overlap, mesa_id):
    # overlap sample of Actigraphy + PSG/HR/Respiration
    row = overlap[overlap["mesaid"] == mesa_id].index
    start_idx = int(overlap["line"][row])
    df_actigraph["line"] = df_actigraph["line"] - start_idx + 1

    # Cut actigraphy data to PSG starting point
    df_actigraph = df_actigraph.truncate(before=start_idx - 1, after=start_idx - 2 + df_psg.size).reset_index(drop=True)

    return df_actigraph[["line", "activity", "linetime"]]
