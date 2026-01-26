import numpy as np
from lime.lime_text import LimeTextExplainer
from distilbert_model_wrapper import predict_proba

class_names = ["not paraphrase", "paraphrase"]
explainer = LimeTextExplainer(class_names=class_names)


def explain_pair(text_a, text_b, num_features=12, num_samples=1500):
    combined = text_a + " [SEP] " + text_b

    def lime_predict(texts):
        pairs = []
        for t in texts:
            if "[SEP]" in t:
                a, b = t.split("[SEP]", 1)
            else:
                a, b = t, ""
            pairs.append((a.strip(), b.strip()))
        return predict_proba(pairs)

    exp = explainer.explain_instance(
        combined,
        lime_predict,
        num_features=num_features,
        num_samples=num_samples
    )

    return dict(exp.as_list())