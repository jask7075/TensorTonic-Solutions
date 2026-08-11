import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
    

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    output = gamma * ((x - mean) / np.sqrt(var + eps)) + beta
    return output

    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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


    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = x @ W1 + b1
    ReLU = np.maximum(0, hidden)
    FFN = ReLU @ W2 + b2

    return FFN



def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    #Multi-head
    attention = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    #Add and norm
    layer_norm1 = layer_norm(x + attention, gamma1, beta1)

    #FFN
    FFN = feed_forward(layer_norm1, W1, b1, W2, b2)

    #Last layer norm
    layer_norm2 = layer_norm(layer_norm1 + FFN, gamma2, beta2)

    return layer_norm2
    