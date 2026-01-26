import numpy as np


def cosine_similarity(v1, v2, eps=1e-8):
    num = np.dot(v1, v2)
    den = (np.linalg.norm(v1) * np.linalg.norm(v2)) + eps
    return num / den
