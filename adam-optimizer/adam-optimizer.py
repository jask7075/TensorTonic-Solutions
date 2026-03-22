import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    #Converting lists to np arrats, element-wise calculations
    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    
    #Updating first and second moment - running averages
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2

    #Bias correction is temporary fix, it doesn't go with the moments.
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    #Calculating updated values for all the params.
    param = param - lr * (m_hat / (v_hat**0.5 + eps))
    return (param, m, v)