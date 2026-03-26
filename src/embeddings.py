import numpy as np

def init_embeddings(vocab_size, embedding_dim=100):
    E = np.random.randn(vocab_size, embedding_dim) * 0.01
    return E

def ids_to_embeddings(E, input_ids):
    input_ids = np.array(input_ids, dtype=np.int64)
    return E[input_ids]
