import pandas as pd


def check_resp_features(df_resp_features):
    if df_resp_features.shape[1] != 61:
        df_resp_features = df_resp_features[
            df_resp_features.columns.drop(list(df_resp_features.filter(regex="270_RRV_DFA")))
        ]
        df_resp_features = df_resp_features[
            df_resp_features.columns.drop(list(df_resp_features.filter(regex="270_RRV_MFDFA")))
        ]
        df_resp_features = df_resp_features[
            df_resp_features.columns.drop(list(df_resp_features.filter(regex="210_RRV_DFA")))
        ]
        df_resp_features = df_resp_features[
            df_resp_features.columns.drop(list(df_resp_features.filter(regex="210_RRV_MFDFA")))
        ]
        df_resp_features = df_resp_features[
            df_resp_features.columns.drop(list(df_resp_features.filter(regex="150_RRV_DFA")))
        ]
        df_resp_features = df_resp_features[
            df_resp_features.columns.drop(list(df_resp_features.filter(regex="150_RRV_MFDFA")))
        ]
        try:
            assert df_resp_features.shape[1] == 61
        except AssertionError:
            print("Number of columns in resp_features is not 61")
    return df_resp_features
