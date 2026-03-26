from src.data_loader import load_data
from src.preprocessing import *
from src.embeddings import *
from src.model import *
from src.utils import *
from src.config import *

def train():
    from tqdm.auto import tqdm

    for epoch in range(epochs):
        epoch_loss = 0.0
        processed_count = 0

        pbar = tqdm(range(len(train_tokenized)), desc=f"Epoch {epoch+1}/{epochs}")

        for i in pbar:
            input_ids = train_tokenized[i]["input_ids"]
            target_ids = train_tokenized[i]["labels"]

            if len(input_ids) == 0 or len(target_ids) == 0:
                continue

            processed_count += 1

            src_sentence = ids_to_embeddings(input_ids)

            # shifted decoder input
            decoder_input_ids, decoder_target_ids = make_decoder_io(target_ids, bos_token_id)
            decoder_input_embs = ids_to_embeddings(decoder_input_ids)

            # Encoder forward
            h_enc = np.zeros((hidden_dim, 1))
            encoder_caches = []

            for t in range(len(src_sentence)):
                x_t = src_sentence[t].reshape(-1, 1)
                h_enc, cache = gru_step_forward(
                    x_t, h_enc,
                    Wzx, Wzh, bz,
                    Wrx, Wrh, br,
                    Whx, Whh, bh
                )
                encoder_caches.append(cache)

            context = h_enc.copy()

            # Decoder forward
            h_dec = context
            decoder_caches = []
            logits_list = []
            probs_list = []

            loss = 0.0

            for t in range(len(decoder_input_embs)):
                x_t = decoder_input_embs[t].reshape(-1, 1)

                h_dec, cache = gru_step_forward(
                    x_t, h_dec,
                    Wzx_dec, Wzh_dec, bz_dec,
                    Wrx_dec, Wrh_dec, br_dec,
                    Whx_dec, Whh_dec, bh_dec
                )

                logits = Wo @ h_dec + bo
                probs = softmax(logits)

                y_true = decoder_target_ids[t]

                if pad_token_id is None or y_true != pad_token_id:
                    loss -= np.log(probs[y_true, 0] + 1e-9)

                decoder_caches.append(cache)
                logits_list.append(logits)
                probs_list.append(probs)

            # Initialize gradients
            dE = np.zeros_like(E)

            dWo = np.zeros_like(Wo)
            dbo = np.zeros_like(bo)

            dWzx_dec = np.zeros_like(Wzx_dec)
            dWzh_dec = np.zeros_like(Wzh_dec)
            dbz_dec = np.zeros_like(bz_dec)

            dWrx_dec = np.zeros_like(Wrx_dec)
            dWrh_dec = np.zeros_like(Wrh_dec)
            dbr_dec = np.zeros_like(br_dec)

            dWhx_dec = np.zeros_like(Whx_dec)
            dWhh_dec = np.zeros_like(Whh_dec)
            dbh_dec = np.zeros_like(bh_dec)

            dWzx = np.zeros_like(Wzx)
            dWzh = np.zeros_like(Wzh)
            dbz = np.zeros_like(bz)

            dWrx = np.zeros_like(Wrx)
            dWrh = np.zeros_like(Wrh)
            dbr = np.zeros_like(br)

            dWhx = np.zeros_like(Whx)
            dWhh = np.zeros_like(Whh)
            dbh = np.zeros_like(bh)

            # Decoder backward
            dh_next = np.zeros((hidden_dim, 1))

            for t in reversed(range(len(decoder_caches))):
                probs = probs_list[t].copy()
                y_true = decoder_target_ids[t]

                if pad_token_id is not None and y_true == pad_token_id:
                    continue

                probs[y_true, 0] -= 1.0

                h_t = decoder_caches[t]["h_t"]

                dWo += probs @ h_t.T
                dbo += probs

                dh = Wo.T @ probs + dh_next

                grads = gru_step_backward(
                    dh,
                    decoder_caches[t],
                    Wzh_dec, Wrh_dec, Whh_dec
                )

                dWzx_dec += grads["dWzx"]
                dWzh_dec += grads["dWzh"]
                dbz_dec += grads["dbz"]

                dWrx_dec += grads["dWrx"]
                dWrh_dec += grads["dWrh"]
                dbr_dec += grads["dbr"]

                dWhx_dec += grads["dWhx"]
                dWhh_dec += grads["dWhh"]
                dbh_dec += grads["dbh"]

                dh_next = grads["dh_prev"]

                token_id = decoder_input_ids[t]
                dE[token_id] += grads["dx"].reshape(-1)

            dh_enc = dh_next

            # Encoder backward
            for t in reversed(range(len(encoder_caches))):
                grads = gru_step_backward(
                    dh_enc,
                    encoder_caches[t],
                    Wzh, Wrh, Whh
                )

                dWzx += grads["dWzx"]
                dWzh += grads["dWzh"]
                dbz += grads["dbz"]

                dWrx += grads["dWrx"]
                dWrh += grads["dWrh"]
                dbr += grads["dbr"]

                dWhx += grads["dWhx"]
                dWhh += grads["dWhh"]
                dbh += grads["dbh"]

                dh_enc = grads["dh_prev"]

                token_id = input_ids[t]
                dE[token_id] += grads["dx"].reshape(-1)

            # L2 Regularization
            dWo += l2_lambda * Wo

            dWzx_dec += l2_lambda * Wzx_dec
            dWzh_dec += l2_lambda * Wzh_dec
            dWrx_dec += l2_lambda * Wrx_dec
            dWrh_dec += l2_lambda * Wrh_dec
            dWhx_dec += l2_lambda * Whx_dec
            dWhh_dec += l2_lambda * Whh_dec

            dWzx += l2_lambda * Wzx
            dWzh += l2_lambda * Wzh
            dWrx += l2_lambda * Wrx
            dWrh += l2_lambda * Wrh
            dWhx += l2_lambda * Whx
            dWhh += l2_lambda * Whh

            dE += l2_lambda * E

            # Gradient clipping
            for grad in [
                dWo, dbo,
                dWzx_dec, dWzh_dec, dbz_dec,
                dWrx_dec, dWrh_dec, dbr_dec,
                dWhx_dec, dWhh_dec, dbh_dec,
                dWzx, dWzh, dbz,
                dWrx, dWrh, dbr,
                dWhx, dWhh, dbh,
                dE
            ]:
                np.clip(grad, -5.0, 5.0, out=grad)

        # SGD + Momentum Update
        vE = momentum * vE - lr * dE
        E += vE

        vWo = momentum * vWo - lr * dWo
        Wo += vWo

        vbo = momentum * vbo - lr * dbo
        bo += vbo

        vWzx_dec = momentum * vWzx_dec - lr * dWzx_dec
        Wzx_dec += vWzx_dec

        vWzh_dec = momentum * vWzh_dec - lr * dWzh_dec
        Wzh_dec += vWzh_dec

        vbz_dec = momentum * vbz_dec - lr * dbz_dec
        bz_dec += vbz_dec

        vWrx_dec = momentum * vWrx_dec - lr * dWrx_dec
        Wrx_dec += vWrx_dec

        vWrh_dec = momentum * vWrh_dec - lr * dWrh_dec
        Wrh_dec += vWrh_dec

        vbr_dec = momentum * vbr_dec - lr * dbr_dec
        br_dec += vbr_dec

        vWhx_dec = momentum * vWhx_dec - lr * dWhx_dec
        Whx_dec += vWhx_dec

        vWhh_dec = momentum * vWhh_dec - lr * dWhh_dec
        Whh_dec += vWhh_dec

        vbh_dec = momentum * vbh_dec - lr * dbh_dec
        bh_dec += vbh_dec

        vWzx = momentum * vWzx - lr * dWzx
        Wzx += vWzx

        vWzh = momentum * vWzh - lr * dWzh
        Wzh += vWzh

        vbz = momentum * vbz - lr * dbz
        bz += vbz

        vWrx = momentum * vWrx - lr * dWrx
        Wrx += vWrx

        vWrh = momentum * vWrh - lr * dWrh
        Wrh += vWrh

        vbr = momentum * vbr - lr * dbr
        br += vbr

        vWhx = momentum * vWhx - lr * dWhx
        Whx += vWhx

        vWhh = momentum * vWhh - lr * dWhh
        Whh += vWhh

        vbh = momentum * vbh - lr * dbh
        bh += vbh

        epoch_loss += loss

        pbar.set_postfix(
            loss=f"{loss:.4f}",
            avg_loss=f"{epoch_loss / processed_count:.4f}",
            lr=f"{lr:.6f}"
        )

    avg_loss = epoch_loss / processed_count
    print(f"epoch: {epoch + 1}, avg_loss: {avg_loss:.4f}, lr: {lr:.6f}")

    if (epoch + 1) % step_size == 0:
        lr = lr * gamma
        print(f"Learning rate updated to: {lr:.6f}")

if __name__ == "__main__":
    train()
