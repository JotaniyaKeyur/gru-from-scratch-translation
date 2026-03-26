def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def make_decoder_io(target_ids, bos_token_id):
    if len(target_ids) < 2:
        return [bos_token_id], target_ids

    decoder_input_ids = [bos_token_id] + target_ids[:-1]
    decoder_target_ids = target_ids[:]
    return decoder_input_ids, decoder_target_ids

def gru_step_forward(x_t, h_prev, Wzx, Wzh, bz, Wrx, Wrh, br, Whx, Whh, bh):
    az = Wzx @ x_t + Wzh @ h_prev + bz
    z = sigmoid(az)

    ar = Wrx @ x_t + Wrh @ h_prev + br
    r = sigmoid(ar)

    ah = Whx @ x_t + Whh @ (r * h_prev) + bh
    h_hat = np.tanh(ah)

    h_t = (1 - z) * h_prev + z * h_hat

    cache = {
        "x_t": x_t,
        "h_prev": h_prev,
        "z": z,
        "r": r,
        "h_hat": h_hat,
        "h_t": h_t,
        "ah": ah
    }
    return h_t, cache

def gru_step_backward(dh_t, cache, Wzh, Wrh, Whh):
    x_t = cache["x_t"]
    h_prev = cache["h_prev"]
    z = cache["z"]
    r = cache["r"]
    h_hat = cache["h_hat"]

    # h_t = (1-z)*h_prev + z*h_hat
    dh_hat = dh_t * z
    dz = dh_t * (h_hat - h_prev)
    dh_prev = dh_t * (1 - z)

    # h_hat = tanh(ah)
    dah = dh_hat * (1 - h_hat**2)

    # ah = Whx x + Whh (r*h_prev) + bh
    dWhx = dah @ x_t.T
    dWhh = dah @ (r * h_prev).T
    dbh = dah

    d_rhprev = Whh.T @ dah
    dr = d_rhprev * h_prev
    dh_prev += d_rhprev * r

    # r = sigmoid(ar)
    dar = dr * r * (1 - r)
    dWrx = dar @ x_t.T
    dWrh = dar @ h_prev.T
    dbr = dar

    dh_prev += Wrh.T @ dar

    # z = sigmoid(az)
    daz = dz * z * (1 - z)
    dWzx = daz @ x_t.T
    dWzh = daz @ h_prev.T
    dbz = daz

    dh_prev += Wzh.T @ daz

    # x gradient
    dx = 0
    dx += Wzx.T @ daz
    dx += Wrx.T @ dar
    dx += Whx.T @ dah

    return {
        "dx": dx,
        "dh_prev": dh_prev,
        "dWzx": dWzx,
        "dWzh": dWzh,
        "dbz": dbz,
        "dWrx": dWrx,
        "dWrh": dWrh,
        "dbr": dbr,
        "dWhx": dWhx,
        "dWhh": dWhh,
        "dbh": dbh
    }
