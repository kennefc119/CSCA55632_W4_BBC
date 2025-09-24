import numpy as np
import sklearn as sk

def matrix_factorization(a):
    """
    - No invalid values are allowed in the input matrix.
    - Performs matrix factorization using Singular Value Decomposition (SVD).
    
    Args:
        a (np.ndarray): Input matrix to be factorized.
        
    Formula : Any_matrix = U * Sigma * V^T
        where Sigma is a diagonal matrix.
        shape of a: m x n
        
        shape of u: m x d
        shape of sigma : d x d
        shape of V transpose: d x n
        """
    k = a.shape[0]
    u, sig, vt = np.linalg.svd(a)
    b = np.matmul(u, np.matmul(np.diag(sig), vt[:k]))

    result = np.allclose(a, b)
    if result:
        status = "Success"
    else:
        status = "Failure"
    return {"result": result,"u": u,"sigma": sig, "vt": vt[:k], "status": status}

