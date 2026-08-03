import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    result = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(d_model//2):
            result[pos][2*i] = np.sin(pos / (10000 ** ((2*i)/d_model)))
            result[pos][2*i + 1] = np.cos(pos / (10000 ** ((2*i)/d_model)))

    return result