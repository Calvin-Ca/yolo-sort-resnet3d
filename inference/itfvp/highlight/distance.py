import numpy as np

def cosin_distance(a, b):
    """
    Cosin distance between two vectors.
    """
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))