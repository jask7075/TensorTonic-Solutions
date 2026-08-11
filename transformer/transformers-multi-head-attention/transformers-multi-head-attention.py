import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    #1. Project Q,K,V
    Q1 = Q @ W_q
    K1 = K @ W_k
    V1 = V @ W_v

    #2. head_size
    d_k = Q.shape[-1] // num_heads

    #3. Scaled dot_products
    batch = Q.shape[0]
    seq_len = Q.shape[1]
    d_model = Q.shape[-1]
    

    #Making sure the last two axes are seq_len and d_k
    #Shape = (batch, num_heads, seq_len, d_k)
    Q_heads = Q1.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K1.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V1.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    multi_head_attention = softmax((Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)) @ V_heads

    multi_head_attention = multi_head_attention.transpose(0, 2, 1, 3)
    multi_head_attention = multi_head_attention.reshape(batch, seq_len, d_model)
    
    return multi_head_attention @ W_o
    


