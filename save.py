import numpy as np


def save_model(path, E, params):

    np.savez(
        path,

        # embeddings
        E=E,

        # encoder
        Wzx=params["Wzx"], Wzh=params["Wzh"], bz=params["bz"],
        Wrx=params["Wrx"], Wrh=params["Wrh"], br=params["br"],
        Whx=params["Whx"], Whh=params["Whh"], bh=params["bh"],

        # decoder
        Wzx_dec=params["Wzx_dec"], Wzh_dec=params["Wzh_dec"], bz_dec=params["bz_dec"],
        Wrx_dec=params["Wrx_dec"], Wrh_dec=params["Wrh_dec"], br_dec=params["br_dec"],
        Whx_dec=params["Whx_dec"], Whh_dec=params["Whh_dec"], bh_dec=params["bh_dec"],

        # output
        Wo=params["Wo"], bo=params["bo"]
    )
