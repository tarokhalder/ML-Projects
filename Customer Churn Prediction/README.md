# Customer Churn Prediction using Machine Learning and Deep Learning

This repository contains a complete pipeline for predicting customer churn using both Artificial Neural Networks (Multi-Layer Perceptron via TensorFlow/Keras) and Machine Learning (Random Forest). The goal is to identify customers who are likely to cancel their subscriptions based on their usage behavior and financial interactions.

## 📊 Dataset Features
The model processes the following customer attributes:
* **Account Info:** Tenure, Subscription Type, Contract Length
* **Usage & Behavior:** Usage Frequency, Support Calls, Last Interaction
* **Financials:** Payment Delay, Total Spend
* **Target:** Churn (1 = Churned, 0 = Retained)

## 🛠️ Project Workflow
1. **Feature Selection:** Dropped irrelevant identifiers (`CustomerID`, `Age`, `Gender`) to prevent bias.
2. **Data Cleaning:** Removed identical duplicate rows.
3. **Outlier Management:** Applied IQR (Interquartile Range) capping/clipping on numerical features to handle extreme values safely without deleting data.
4. **Categorical Encoding:** Converted text fields into numerical data using One-Hot Encoding (`pd.get_dummies`).
5. **Feature Scaling:** Standardized the dynamic ranges using `StandardScaler` for optimized neural network convergence.
6. **Class Imbalance Handling:** Addressed data imbalance using calculated class weights.

## 🧠 Models Implemented
* **Deep Learning:** Multi-Layer Perceptron (MLP) built with TensorFlow/Keras (`Sequential`, `Dense` layers, `Adam` optimizer).

## 📈 Performance Summary
* **TensorFlow MLP Accuracy:** 57.81%
* **Random Forest Benchmarking:** Implemented as a comparative step to analyze structured data limits.

## 💻 Technical Stack
* **Language:** Python
* **Libraries:** TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, Matplotlib
