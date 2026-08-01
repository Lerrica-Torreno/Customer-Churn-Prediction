Excellentimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# This script performs Telco customer churn prediction and visualizes model performance.

# Data Loading and Preprocessing
# Loads the dataset, handles missing values, and encodes categorical features for model compatibility.
try:
    # NOTE: You'll need to make sure this path is correct on your machine.
    df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\WA_Fn-UseC_-Telco-Customer-Churn.csv") 
except FileNotFoundError:
    print("Error: The CSV file was not found.")
    exit()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop("customerID", axis=1)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Data Splitting
# Splits the dataset into features (X) and the target variable (y). The split is stratified to maintain the class distribution in both training and testing sets.
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Model Training
# Trains a Decision Tree classifier and a K-Nearest Neighbors (KNN) classifier. Data is scaled using StandardScaler prior to KNN training to ensure uniform feature influence.
dt = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Performance Metric Calculation
# Calculates and stores key performance metrics (confusion matrix, accuracy, F1 score, precision, and recall) for both models.
labels = ["No", "Yes"]

cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_knn = confusion_matrix(y_test, y_pred_knn)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)

data = {
    'Decision Tree': {
        'cm': cm_dt,
        'accuracy': accuracy_dt,
        'precision': precision_dt,
        'recall': recall_dt
    },
    'KNN': {
        'cm': cm_knn,
        'accuracy': accuracy_knn,
        'precision': precision_knn,
        'recall': recall_knn
    }
}

# Performance Visualization
# Generates a visual comparison of the model's performance. A table-based plot displays the confusion matrix, accuracy, precision, and recall for each classifier.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Comparison', fontsize=16)

# --- CORRECTED FUNCTION START ---
# The function now accepts 'y_pred_model' to correctly calculate class-wise metrics.
def plot_performance_table(ax, model_name, cm, accuracy, y_pred_model):
    table_data = [
        ['', 'true No', 'true Yes'],
        ['pred No', cm[0, 0], cm[0, 1]],
        ['pred Yes', cm[1, 0], cm[1, 1]],
        ['class precision', '', ''],
        ['class recall', '', '']
    ]

    # Use y_pred_model for correct class precision/recall calculation
    # 'true No' (pos_label=0)
    table_data[3][1] = f'{precision_score(y_test, y_pred_model, pos_label=0):.2%}'
    table_data[4][1] = f'{recall_score(y_test, y_pred_model, pos_label=0):.2%}'
    
    # 'true Yes' (default pos_label=1)
    table_data[3][2] = f'{precision_score(y_test, y_pred_model):.2%}'
    table_data[4][2] = f'{recall_score(y_test, y_pred_model):.2%}'

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax.set_title(f'{model_name}\nAccuracy: {accuracy:.2%}', fontsize=12)
    ax.axis('off')

# --- CORRECTED FUNCTION END ---

# --- CORRECTED FUNCTION CALLS START ---
# Pass the respective prediction arrays (y_pred_dt and y_pred_knn)
plot_performance_table(axes[0], 'Decision Tree', data['Decision Tree']['cm'], 
                    data['Decision Tree']['accuracy'], 
                    y_pred_dt) # Pass y_pred_dt
                    
plot_performance_table(axes[1], 'K-Nearest Neighbors', data['KNN']['cm'], 
                    data['KNN']['accuracy'], 
                    y_pred_knn) # Pass y_pred_knn

# --- CORRECTED FUNCTION CALLS END ---

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Detailed Metrics Output
# Prints the detailed classification reports and confusion matrices to the console for in-depth analysis of model performance.
print("Decision Tree Results:")
print(f"Accuracy: {accuracy_dt:.2%}")
print(f"F1 Score: {f1_dt:.2%}")
print(f"Precision: {precision_dt:.2%}")
print(f"Recall: {recall_dt:.2%}")
print("\nConfusion Matrix:\n", pd.DataFrame(cm_dt, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels]))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt, target_names=labels))


print("\nKNN Results:")
print(f"Accuracy: {accuracy_knn:.2%}")
print(f"F1 Score: {f1_knn:.2%}")
print(f"Precision: {precision_knn:.2%}")
print(f"Recall: {recall_knn:.2%}")
print("\nConfusion Matrix:\n", pd.DataFrame(cm_knn, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels]))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn, target_names=labels))