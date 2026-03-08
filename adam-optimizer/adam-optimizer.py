import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    # 1. Mise à jour des moments (moyennes mobiles)
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    
    # 2. Correction du biais (Notez l'exposant **t)
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    
    # 3. Mise à jour du paramètre
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # 4. Retour conforme à la docstring : (param, m, v)
    return param_new, m_new, v_new