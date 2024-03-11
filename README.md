# predictive_maintenance

This study compares four popular supervised machine learning models to predict machine failures for predictive maintenance applications:
* Logistic regression
* K nearest neighbor
* Support vector machine
* Decision trees

A comprehensive comparison is conducted, assessing each model's accuracy, F1 score, training, and test times. Additionally, the benefit and cost of using hyperparameter optimization is investigated.

The evaluation uses a realistic and synthetic predictive maintenance dataset published to the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), comprising of 14 dimensions across 10,000 samples.
* Identifiers (2): UID, Product ID
* Input parameters (6): Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
* Response (1): Machine failure
* Failure modes (5): TWF (Tool Wear Failure), HDF (Heat Dissipation Failure), PWF (Power Failure), OSF (Overstrain Failure), RNF (Random Failures)

Among the models examined, the decision tree emerges as the top performer, achieving an F1 score of 0.74 and accuracy of 0.98 while exhibiting comparable training times to the other three models.

| Model          | F1 Score | Precision | Recall | MCC  | Accuracy | CV F1 Score | Overkill | Escape | Training Time | Test Time | Memory Usage |
|-----------------|----------|-----------|--------|------|----------|-------------|----------|--------|----------------|-----------|--------------|
| DT_Optimized    | 0.735    | 0.735     | 0.735  | 0.726| 0.982    | 0.692       | 0.009    | 0.009  | 6.584          | 0.000     | 0.000        |
| DT_Default      | 0.711    | 0.716     | 0.706  | 0.701| 0.980    | 0.639       | 0.010    | 0.010  | 0.133          | 0.001     | 0.031        |
| SVM_Optimized   | 0.608    | 0.912     | 0.456  | 0.637| 0.980    | 0.608       | 0.002    | 0.018  | 9.954          | 0.043     | 12.391       |
| KNN_Optimized   | 0.485    | 0.714     | 0.368  | 0.501| 0.974    | 0.508       | 0.005    | 0.022  | 2.786          | 0.041     | 0.125        |
| KNN_Default     | 0.468    | 0.846     | 0.324  | 0.514| 0.975    | 0.432       | 0.002    | 0.023  | 0.113          | 0.036     | 0.000        |
| SVM_Default     | 0.440    | 0.870     | 0.294  | 0.497| 0.974    | 0.336       | 0.002    | 0.024  | 0.219          | 0.049     | 11.953       |
| LR_Default      | 0.276    | 0.632     | 0.176  | 0.323| 0.968    | 0.307       | 0.004    | 0.028  | 0.130          | 0.000     | 0.750        |
| LR_Optimized    | 0.273    | 0.600     | 0.176  | 0.314| 0.968    | 0.315       | 0.004    | 0.028  | 3.722          | 0.000     | 7.562        |

<br>

These findings underscore the potential of machine learning techniques, particularly decision trees, in enabling predictive maintenance strategies, contingent upon the availability of relevant and high-quality input data.

[Click here](https://github.com/yuzom/predictive_maintenance/blob/main/predictive_maintenance.ipynb) to see the code in action.

**(C) 2024 Yuzo Makitani** This repository is released under the [GNU GPLv3.0 or later](https://www.gnu.org/licenses/).
