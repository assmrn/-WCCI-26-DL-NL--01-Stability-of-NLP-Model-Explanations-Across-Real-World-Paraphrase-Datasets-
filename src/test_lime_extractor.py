from lime_explanation_utils import explain_pair

a = "How can I learn machine learning?"
b = "What is the best way to study machine learning?"

exp = explain_pair(a, b)

print("\nLIME explanation dictionary:\n")
for k, v in exp.items():
    print(f"{k:>12s} : {v:.4f}")
