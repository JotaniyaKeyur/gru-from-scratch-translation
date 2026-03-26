import numpy as np


def init_model(vocab_size, embedding_dim=100, hidden_dim=100):

    input_dim = embedding_dim
    std = np.sqrt(2.0 / (input_dim + hidden_dim))

    Wzx = np.random.randn(hidden_dim, input_dim) * std
    Wzh = np.random.randn(hidden_dim, hidden_dim) * std
    bz = np.zeros((hidden_dim, 1))

    Wrx = np.random.randn(hidden_dim, input_dim) * std
    Wrh = np.random.randn(hidden_dim, hidden_dim) * std
    br = np.zeros((hidden_dim, 1))

    Whx = np.random.randn(hidden_dim, input_dim) * std
    Whh = np.random.randn(hidden_dim, hidden_dim) * std
    bh = np.zeros((hidden_dim, 1))

    Wzx_dec = np.random.randn(hidden_dim, input_dim) * std
    Wzh_dec = np.random.randn(hidden_dim, hidden_dim) * std
    bz_dec = np.zeros((hidden_dim, 1))

    Wrx_dec = np.random.randn(hidden_dim, input_dim) * std
    Wrh_dec = np.random.randn(hidden_dim, hidden_dim) * std
    br_dec = np.zeros((hidden_dim, 1))

    Whx_dec = np.random.randn(hidden_dim, input_dim) * std
    Whh_dec = np.random.randn(hidden_dim, hidden_dim) * std
    bh_dec = np.zeros((hidden_dim, 1))

    Wo = np.random.randn(vocab_size, hidden_dim) * 0.01
    bo = np.zeros((vocab_size, 1))

    return {
        # encoder
        "Wzx": Wzx, "Wzh": Wzh, "bz": bz,
        "Wrx": Wrx, "Wrh": Wrh, "br": br,
        "Whx": Whx, "Whh": Whh, "bh": bh,

        # decoder
        "Wzx_dec": Wzx_dec, "Wzh_dec": Wzh_dec, "bz_dec": bz_dec,
        "Wrx_dec": Wrx_dec, "Wrh_dec": Wrh_dec, "br_dec": br_dec,
        "Whx_dec": Whx_dec, "Whh_dec": Whh_dec, "bh_dec": bh_dec,

        # output
        "Wo": Wo, "bo": bo
    }


def init_velocities(params):
    """
    Initializes momentum velocities for all parameters.
    """
    velocities = {}
    for key in params:
        velocities["v" + key] = np.zeros_like(params[key])
    return velocities
