# Customer Churn Prediction Using Classification Techniques

A machine learning project that predicts whether a telecommunications customer is likely to discontinue their subscription.

The project uses the **Telco Customer Churn Dataset** and compares different classification algorithms using **Python** and **RapidMiner AI Studio**.

## Project Overview

Customer churn occurs when a customer stops using a company's products or services. Predicting customer churn can help telecommunications companies identify customers who may leave and take proactive steps to retain them.

This project evaluates the following classification models:

- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Random Forest Classifier

The primary goal is to build an interpretable classification model that can identify customers who are at risk of churning.

## Objectives

- Prepare and preprocess the Telco Customer Churn Dataset.
- Train classification models for churn prediction.
- Compare the performance of Decision Tree, KNN, and Random Forest.
- Identify the customer attributes that influence churn.
- Provide insights that may support customer-retention strategies.

## Technologies Used

- Python
- RapidMiner AI Studio
- Visual Studio Code
- Pandas
- Scikit-learn
- Matplotlib
- CSV

## Machine Learning Models

### Decision Tree

The Decision Tree Classifier was selected as the primary model because its decision rules are easier to interpret and visualize.

For the Python implementation, the model used:

- Criterion: `entropy`
- Maximum depth: `5`
- Training and testing split: `70% / 30%`
- Stratified sampling

Limiting the maximum depth helped maintain interpretability and reduce the possibility of overfitting.

### K-Nearest Neighbors

The KNN Classifier was used as a comparison model.

The implementation used:

- Number of neighbors: `5`
- Standardized numerical features
- Training and testing split: `70% / 30%`

Feature scaling was applied because KNN is a distance-based classification algorithm.

### Random Forest

Random Forest was also tested in RapidMiner. It achieved slightly higher overall accuracy than the Decision Tree, but its predictions were more difficult to explain through clear decision rules.

## Data Preprocessing

The dataset underwent several preprocessing steps before model training:

1. Imported the Telco Customer Churn Dataset from a CSV file.
2. Removed attributes considered irrelevant to the prediction process.
3. Handled missing values.
4. Converted categorical attributes into numerical values.
5. Separated the dataset into features and the target variable.
6. Divided the data into 70% training data and 30% testing data.
7. Used stratified sampling to preserve the proportion of churn and non-churn records.
8. Applied `StandardScaler` to the data used by the KNN model.

The `Churn` attribute was used as the target variable.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Recall for the churn class was considered especially important because it measures how many actual churners were correctly identified.

## Results

### RapidMiner Results

| Model | Accuracy |
|---|---:|
| Decision Tree | 79.03% |
| Random Forest | 80.45% |

The RapidMiner Decision Tree achieved an F1 score of **57.28%** for the churn class.

Although Random Forest achieved slightly higher accuracy, the Decision Tree was easier to interpret and visualize.

### Python Results

| Model | Accuracy | Churn Precision | Churn Recall |
|---|---:|---:|---:|
| Decision Tree | 78.70% | 60.04% | 59.18% |
| K-Nearest Neighbors | 74.35% | 51.74% | 50.27% |

The Decision Tree performed better than KNN in overall accuracy and in identifying customers who actually churned.

## Important Findings

The most significant predictors identified by the models were:

- Tenure
- Monthly charges
- Total charges
- Senior-citizen status

The analysis suggests that:

- New customers are more likely to churn.
- Customers with high monthly charges have a greater risk of leaving.
- Customers with longer tenure are less likely to churn.
- Long-term customers with high total charges are more likely to remain subscribed.

These findings may help a telecommunications company identify high-risk customers and develop targeted retention programs.

## Business Applications

The model could support customer-retention efforts such as:

- Offering discounts to high-risk customers.
- Creating loyalty programs for new subscribers.
- Providing personalized customer support.
- Reviewing plans with high monthly charges.
- Improving the customer experience during the first few months of subscription.
- Supporting proactive customer relationship management.

## Limitations

- The dataset contains more non-churn customers than churn customers.
- The class imbalance affects the models' ability to identify all churn cases.
- Some potential predictors, such as customer-satisfaction scores, are not included.
- The results are based on a telecommunications dataset and may not directly apply to other industries.
- The models still produce false negatives, meaning some actual churners are not detected.

## Future Improvements

Future versions of the project may include:

- Applying SMOTE or other class-balancing techniques.
- Performing hyperparameter tuning.
- Testing additional models such as Logistic Regression, XGBoost, or Support Vector Machines.
- Using cross-validation for more reliable evaluation.
- Adding ROC curves and precision-recall curves.
- Deploying the selected model through a web application.
- Developing an interactive dashboard for churn analysis.
- Adding customer-satisfaction and service-quality variables.

## How to Run the Python Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

For Windows:

venv\Scripts\activate

For macOS or Linux:

source venv/bin/activate
4. Install the required libraries
pip install pandas numpy scikit-learn matplotlib
5. Add the dataset

Place the Telco Customer Churn CSV file inside the project folder or update the dataset path in the Python script.

6. Run the program
python main.py

Replace main.py with the actual name of the Python file in the repository.

Authors
Lerrica Jeremy S. Torreno
Christian Joshua C. Manaog
Date

September 28, 2025

Acknowledgments
Telco Customer Churn Dataset
Altair RapidMiner AI Studio
Scikit-learn
Pandas
Disclaimer

This project was developed for academic and educational purposes. The predictions should not be used as the sole basis for real-world business decisions without additional validation.


The README reflects the project’s documented preprocessing process, model settings, results, key predictors
