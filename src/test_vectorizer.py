from lime_explanation_utils import explain_pair
from explanation_vectorizer import vectorize_explanations

a1 = "How can I learn machine learning?"
b1 = "What is the best way to study machine learning?"

a2 = "How do I start learning machine learning?"
b2 = "What is the best way to study machine learning?"

exp1 = explain_pair(a1, b1)
exp2 = explain_pair(a2, b2)

features, v1, v2 = vectorize_explanations(exp1, exp2)

print("\nFeatures:")
print(features)

print("\nVector 1:")
print(v1)

print("\nVector 2:")
print(v2)
