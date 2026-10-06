"""
compare_models.py
Loads both models' evaluation reports and prints a comparison table.
"""

import json

with open("models/evaluation_reports.json", "r") as f:
    reports = json.load(f)

cnn = reports["cnn"]
transfer = reports["transfer"]

print(f"{'Model':<20}{'Accuracy':<12}{'Macro Precision':<18}{'Macro Recall':<15}{'Macro F1':<10}")
print("-" * 75)
print(f"{'CNN Baseline':<20}{cnn['accuracy']:<12.4f}{cnn['macro avg']['precision']:<18.4f}"
      f"{cnn['macro avg']['recall']:<15.4f}{cnn['macro avg']['f1-score']:<10.4f}")
print(f"{'Transfer Learning':<20}{transfer['accuracy']:<12.4f}{transfer['macro avg']['precision']:<18.4f}"
      f"{transfer['macro avg']['recall']:<15.4f}{transfer['macro avg']['f1-score']:<10.4f}")

print("\nPer-class F1-score comparison:")
print(f"{'Class':<12}{'CNN':<10}{'Transfer':<10}")
for class_name in cnn:
    if class_name not in ("accuracy", "macro avg", "weighted avg"):
        print(f"{class_name:<12}{cnn[class_name]['f1-score']:<10.4f}{transfer[class_name]['f1-score']:<10.4f}")