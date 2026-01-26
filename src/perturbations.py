import re

def simple_perturb(text: str):
    """
    Light semantic-preserving perturbation:
    - lowercase
    - remove punctuation
    - normalize spaces
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
