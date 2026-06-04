# Models Comparison

| Model | Accuracy | ROC-AUC | F1 | Precision | Recall | Inference latency (ms) |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.9333 | 0.9876 | 0.9009 | 0.8708 | 0.9330 | 3.01 |
| Random Forest | 0.9696 | 0.9965 | 0.9522 | 0.9721 | 0.9330 | 103.43 |
| Gradient Boosting | 0.9710 | 0.9977 | 0.9543 | 0.9766 | 0.9330 | 6.46 |


Gradient Boosting achieved the strongest overall performance, with the highest accuracy, ROC-AUC, F1-score and precision. Random Forest performed very similarly and clearly improved over the Logistic Regression baseline. Logistic Regression remains useful as a simple baseline model, but tree-based models fit this dataset better.
