import numpy as np


def vectorize_explanations(exp1: dict, exp2: dict):
    """
    Takes two LIME explanation dictionaries
    Returns aligned vectors and the feature list
    """

    features = sorted(set(exp1.keys()).union(set(exp2.keys())))

    v1 = np.array([exp1.get(f, 0.0) for f in features])
    v2 = np.array([exp2.get(f, 0.0) for f in features])

    return features, v1, v2
