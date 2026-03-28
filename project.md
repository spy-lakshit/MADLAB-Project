Project 1: Intelligent Hardware Thermal Failure Prediction System
(Classification-Based System)

Problem Statement: Data centers and high-performance computing setups frequently suffer from unexpected hardware failures due to thermal throttling and voltage irregularities. Relying on simple temperature thresholds is ineffective because failures often result from complex combinations of CPU/GPU loads, fan speeds, and voltage variances.

Objective: To build an intelligent predictive maintenance system that classifies the immediate failure risk of computer hardware, allowing for preemptive cooling or workload redistribution.

Algorithms Used & Justification:

Decision Tree (ID3/CART): Used as the primary interpretable model. Hardware technicians need to understand why a system is flagged for failure (e.g., "If GPU Temp > 85°C AND Fan RPM < 1200 -> High Risk").

Artificial Neural Network (Backpropagation): Used as the high-accuracy secondary model. An ANN can detect complex, non-linear relationships between variables (like subtle voltage drops combined with memory load) that a Decision Tree might miss.

Dataset Structure (Sample CSV):

CPU_Temp (numeric, Celsius)

GPU_Temp (numeric, Celsius)

Fan_RPM (numeric)

Voltage_Variance (numeric, Volts)

RAM_Usage (numeric, Percentage)

Target_Risk (categorical: 0 = Safe, 1 = Warning, 2 = Critical)

Complete Python Code:

Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Generate Synthetic Hardware Telemetry Data (Replace with real CSV if available)
np.random.seed(42)
n_samples = 1000

data = {
    'CPU_Temp': np.random.normal(65, 15, n_samples),
    'GPU_Temp': np.random.normal(70, 20, n_samples),
    'Fan_RPM': np.random.normal(2000, 500, n_samples),
    'Voltage_Variance': np.random.uniform(0.01, 0.15, n_samples),
    'RAM_Usage': np.random.uniform(30, 95, n_samples)
}
df = pd.DataFrame(data)

# Define risk based on a hidden complex rule (to test the algorithms)
def assign_risk(row):
    if row['CPU_Temp'] > 85 and row['Fan_RPM'] < 1500:
        return 2 # Critical
    elif row['GPU_Temp'] > 80 or row['Voltage_Variance'] > 0.1:
        return 1 # Warning
    else:
        return 0 # Safe

df['Target_Risk'] = df.apply(assign_risk, axis=1)

# 2. Data Preprocessing
X = df.drop('Target_Risk', axis=1)
y = df['Target_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Decision Tree
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5) # Entropy matches ID3 concept
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# 4. Train Artificial Neural Network (Backpropagation)
# Hidden layers: 2 layers of 16 and 8 nodes. Solver: adam (optimized backprop)
ann_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)
ann_preds = ann_model.predict(X_test)

# 5. Evaluation
print("--- Decision Tree Performance ---")
print(classification_report(y_test, dt_preds))

print("\n--- ANN (Backpropagation) Performance ---")
print(classification_report(y_test, ann_preds))
Step-by-Step Explanation:

Data Generation: We simulate telemetry data from computer components. In a real scenario, you would use pd.read_csv('hardware_logs.csv').

Preprocessing: We split the features (sensors) from the target (risk level) and divide the data into 80% training and 20% testing sets.

Decision Tree: We initialize the classifier using entropy to mimic the ID3 information gain logic you learned. It builds conditional splits.

ANN: We initialize a Multi-Layer Perceptron. The fit method uses backpropagation to adjust the weights across the hidden layers to map sensor data to risk levels.

Evaluation: We output precision, recall, and accuracy to compare which model handled the dataset better.

Output Explanation & Metrics:
The output prints two classification reports.

Accuracy: Overall correctness.

Precision: Out of all instances the model predicted as "Critical", how many were actually critical? (Crucial to avoid false alarms).

Recall: Out of all actual "Critical" instances, how many did the model catch? (Crucial to avoid hardware burning out).

Advantages & Limitations:

Advantage: Combines the "black box" high accuracy of an ANN with the transparent, rule-based logic of a Decision Tree.

Limitation: ANNs require significant data to train without overfitting and are computationally heavier to deploy on low-power IoT edge devices.

Future Scope:
Integrating a time-series element (like an LSTM) to predict when exactly the failure will occur in the future, rather than just current risk.

Viva Questions & Answers:

Q: Why did you use entropy in your Decision Tree?

A: Entropy measures the impurity of a dataset. By calculating Information Gain (which relies on entropy), the ID3 algorithm decides which hardware sensor splits the data most cleanly into safe vs. critical states.

Q: How does backpropagation work in your ANN?

A: It calculates the error (difference between predicted risk and actual risk) at the output layer, then propagates this error backward through the network, updating the weights using an optimization algorithm (like Adam or Gradient Descent) to minimize the loss.
