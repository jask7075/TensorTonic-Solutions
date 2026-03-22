import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    #Converting list into numpy arrays
    #Numpy array calculations work element-wise
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    #Update second moment
    s = beta * s + (1-beta) * g**2
    #Update params/ weights
    w = w - lr * g / (s + eps)**0.5

    return (w, s)