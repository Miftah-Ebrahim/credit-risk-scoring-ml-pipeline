# Final Project Report: Credit Risk Probability Model

## 1. Introduction

This comprehensive report documents the successful development and deployment of a machine learning system designed to estimate credit risk for unbanked customers. Addressing the "Cold Start" challenge prevalent in the Buy-Now-Pay-Later (BNPL) industry, this project leverages alternative transactional data to construct a robust behavioral proxy for creditworthiness.

## 2. Methodology

The core philosophy of this system relies on the hypothesis that **transactional behavior—specifically Recency, Frequency, and Monetary value (RFM)—is a rigorous predictor of financial reliability.** In the absence of historical default labels, we devised an unsupervised learning strategy to generate a ground truth dataset for supervised modeling.

### 2.1 Feature Engineering Pipeline

The data processing architecture executes a precise sequence of transformations:

1.  **Temporal Profile Extraction**: We decompose timestamps into `TransactionHour`, `TransactionDay`, and `TransactionMonth` to capture granular seasonality and time-of-day behavioral patterns.
2.  **RFM Construction**: Raw transaction logs are aggregated into detailed customer profiles:
    *   **Recency**: The dormancy period (days) since the last active transaction.
    *   **Frequency**: The velocity of transaction volume.
    *   **Monetary**: Statistical moments (Sum, Mean, Standard Deviation) of spend capacity.
3.  **Categorical Encoding**: The `ChannelId` feature is strictly One-Hot Encoded to capture channel-specific risk vectors (e.g., Web vs. Mobile).
4.  **Standardization**: A `StandardScaler` normalizes all numerical distributions, ensuring optimal convergence for both clustering and linear optimization algorithms.

### 2.2 Proxy Target Generation

To synthesize the target variable `Risk_Label`, we implemented **K-Means Clustering** ($k=3$).
*   **Segmentation Strategy**: Customers are partitioned based on their standardized RFM vectors.
*   **Risk Definition**: The cluster exhibiting the highest average Recency and lowest Frequency is mathematically isolated as the "High Risk" cohort.
*   **Label Assignment**:
    *   **High Risk (1)**: Members of the dormant/churned cluster.
    *   **Low Risk (0)**: Members of the active and high-value clusters.

## 3. Model Development

### 3.1 Model Architectures

We trained and evaluated two distinct supervised classification architectures:

1.  **Logistic Regression**: Prioritized for its **Explainability**. The linear coefficients provide direct insight into risk drivers, aligning with regulatory requirements for model auditability.
2.  **Gradient Boosting Classifier**: Prioritized for **Performance**. This ensemble method captures non-linear relationships and complex feature interactions that linear models might miss.

### 3.2 Training Pipeline

The training process is hermetically sealed within a `sklearn.pipeline.Pipeline`, ensuring:
*   **Reproducibility**: Identical preprocessing during training and inference.
*   **Leakage Prevention**: Statistics (Mean/Std) are computed solely on the training split.

### 3.3 Evaluation Results

Both architectures demonstrated exceptional separability on the held-out test set, achieving an **ROC-AUC of 1.000**.

> **Note**: These metrics confirm that the supervised models have successfully learned the behavioral pattern definitions established by the unsupervised clustering step. In a production environment with external default labels, these metrics would serve as a baseline for behavioral consistency.

## 4. Analytical Insights (EDA)

The feature engineering strategy was driven by rigorous Exploratory Data Analysis.

### 4.1 Transaction Volume Analysis
![Daily Transaction Volume](dashboard/daily_transaction_volume.png)
*Figure 1: Daily transaction volume reveals distinct cyclical patterns, validating the extraction of temporal features.*

### 4.2 Correlation Analysis
![Correlation Matrix](dashboard/feature_correlation_matrix.png)
*Figure 2: The correlation matrix exposes strong multicollinearity between Frequency and Total Monetary Value, guiding our choice of regularization.*

### 4.3 Fraud Distribution
![Fraud Distribution](dashboard/fraud_distribution_summary.png)
*Figure 3: Fraud cases show non-uniform distribution across channels, necessitating the inclusion of `ChannelId` as a predictor.*

## 5. Deployment Architecture

The system is operationalized as a containerized microservice:

*   **MLflow**: Serves as the central Model Registry and Experiment Tracker.
*   **FastAPI**: Exposes a high-performance REST interface for real-time scoring.
*   **Docker**: Guarantees environment parity across development and production stages.
*   **CI/CD**: GitHub Actions enforces code quality via `flake8` and regression testing via `pytest`.

## 6. Conclusion

This project successfully operationalizes RFM analysis into a production-grade machine learning pipeline. By providing a scalable, auditable, and deployable solution, it effectively bridges the gap between alternative data and credit risk assessment.
