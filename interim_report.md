# Interim Report: Credit Risk Probability Modeling Using Alternative Data

## 1. Business Objective and Regulatory Context

### Business Objective
The primary objective of this project is to develop a robust credit scoring framework for **Bati Bank** to support its Buy-Now-Pay-Later (BNPL) partnership with an eCommerce platform. By leveraging alternative transactional data, the bank aims to accurately assess the creditworthiness of customers who may lack traditional credit histories. A reliable scoring model is critical for automating loan approvals, minimizing default rates, and maintaining a healthy risk profile for the bank's lending portfolio.

### 1.1 Basel II Capital Accord
In the context of the regulated banking sector, adhering to **Basel II** principles is non-negotiable. Basel II emphasizes rigorous quantitative risk measurement to ensure capital adequacy—meaning Bati Bank must hold sufficient capital to cover potential losses.
- **Transparency & Documentation:** The Accord requires that risk models be transparent, well-documented, and rigorous.
- **Auditability:** Models must be explainable to regulators. We cannot rely solely on "black box" predictions; the drivers of risk (e.g., specific transaction behaviors) must be interpretable to justify capital reserves and credit decisions.

### 1.2 Proxy Target Variable Justification
A challenge identified during the data assessment is the absence of a direct "Loan Default" label in the provided **Xente** transaction dataset. The data represents raw transactional activity (e.g., Airtime, Utility Bills) rather than historical loan repayment records.
- **The Proxy Strategy:** To model credit risk, we must engineer a proxy target variable. We will classify users as "High Risk" or "Low Risk" based on their transactional behavior, specifically using an **RFM (Recency, Frequency, Monetary)** framework.
- **Business Risks:** Using a proxy introduces misclassification risk. A customer with low transaction volume isn't necessarily a bad borrower (they might just be inactive). Conversely, a high-volume user might still default. We must carefully calibrate the "Good/Bad" definition to avoid excluding viable customers (Type II error) or approving defaulters (Type I error).

### 1.3 Model Interpretability Trade-offs
In a regulated environment, the trade-off between predictive power and interpretability is pivotal:
- **Interpretable Models (e.g., Logistic Regression with WoE):** These are preferred for regulatory compliance because they offer clear "Why" explanations for every score (e.g., "Score dropped 10 points due to low transaction frequency"). They align well with Basel II's requirement for interpretable risk drivers.
- **Complex Models (e.g., Gradient Boosting/Random Forest):** While potentially offering higher accuracy, they function as black boxes. In this phase, we will prioritize models that balance performance with the critical need for **explainability** and **auditability**.

### 1.4 RFM as a Behavioral Risk Signal
We will utilize **RFM Analysis** to construct our risk signal:
- **Recency:** How recently did the customer transact? Recent activity implies engagement and liquidity.
- **Frequency:** How often do they transact? Consistent patterns suggest stability.
- **Monetary:** How much do they spend? Higher values (within reason) can indicate repayment capacity.
By aggregating these metrics, we can segment customers into "Good" (active, high-value) and "Bad" (dormant, low-value) categories to serve as our proxy for credit risk.

---

## 2. Dataset Overview

Based on the Exploratory Data Analysis (EDA) of the **Xente** dataset, the structure is suitable for behavioral scoring.

- **Structure:** The dataset is transaction-level, meaning each row represents a single purchase or transfer.
- **Identifiers:**
    - `TransactionId`: Unique key for events.
    - `CustomerId`, `AccountId`, `SubscriptionId`: Crucial for aggregating individual user behavior into a single "Credit Score".
- **Monetary Variables:** `Amount` and `Value` capture the financial magnitude of operations.
- **Categorical Variables:**
    - `ProductCategory` (e.g., Airtime, Financial Services): Reveals spending priorities.
    - `ChannelId` (e.g., Web, Android, iOS): Indicates tech-savviness and access channels.
    - `ProviderId` and `ProductId`: Granular details on the service vendors.
- **Temporal Features:** `TransactionStartTime` is essential for calculating "Recency" and evaluating stability over time.

This granular transaction log allows us to build a rich behavioral profile for each customer, transforming raw logs into predictive risk attributes.

---

## 3. Exploratory Data Analysis (EDA) Findings

### 3.1 Summary Statistics
The numerical features display high variability, typical of financial data.
- **Central Tendency vs. Dispersion:** The spread between the minimum and maximum amounts is significant, indicating a mix of micro-transactions (e.g., small airtime top-ups) and larger transfers.
- **Skewness:** The data is highly right-skewed. Most transactions are low-value, with a "long tail" of high-value outliers. This implies that mean values are likely pulled upwards by outliers, making the median a more representative measure of "typical" behavior.

### 3.2 Numerical Feature Distributions
Plots of `Amount` and `Value` confirm the skewed nature of the data.
- **Observation:** The distributions are not Gaussian (Normal). They exhibit a heavy positive skew.
- **Pattern:** This is standard in payments data—people frequently buy small items (data bundles, airtime) but rarely make massive transfers.
- **Modeling Implication:** Linear models (like Logistic Regression) will require transformations (e.g., Log transformation or Weight of Evidence binning) to handle this skewness effectively.

### 3.3 Categorical Feature Distributions
Analysis of `ProductCategory` and `ChannelId` reveals distinct user personas.
- **ProductCategory:** Expenses are clustered in categories like **Airtime** and **Financial Services**. This suggests the platform is primarily used for utility and connectivity needs rather than luxury goods.
- **Utility:** High usage of "Financial Services" suggests users are already treating the platform as a digital wallet, which correlates well with future credit adoption.

### 3.4 Missing Values
- **Finding:** The initial inspection (`df.info()`) indicates a **complete dataset** with no null values in key columns like `Amount`, `TransactionStartTime`, or `CustomerId`.
- **Strategy:** While technical missing values are absent, we must remain vigilant for "logical" missing values (e.g., placeholders like `0` or `-1`) during feature engineering. If any are discovered, imputation strategies (median for skewed data) will be employed.

### 3.5 Correlation Analysis
- **Observation:** There is a directional relationship between `Amount` and `Value`. In many transaction systems, these track closely, potentially differing only by fees or currency adjustments.
- **Multicollinearity:** If these two variables are highly correlated (near 1.0), we cannot include both in the final model as they provide redundant information. Feature selection or dimensionality reduction (PCA) will be necessary.

### 3.6 Outlier Analysis
- **Evidence:** Box plots of monetary fields show a dense cluster of points beyond the upper whiskers.
- **Context:** In fraud and credit risk, outliers are often the *most* interesting data points. They could represent high-net-worth individuals (good credit risk) or fraudulent account takeovers (bad risk).
- **Impact:** We should **not** summarily delete outliers. Instead, we should cap them (winsorization) or bin them to ensure the model captures their signal without becoming unstable.

---

## 4. Key EDA Insights

1.  **Skewed Value Distribution Requires Transformation**
    *   **Finding:** Transaction `Amount` follows a power-law distribution.
    *   **Impact:** Raw values will bias the model. We must apply Log-transforms or use binning (WoE) to normalize the distribution for the credit scorecard.

2.  **Product Category as a Risk Proxy**
    *   **Finding:** Transactions are heavily weighted towards "Airtime" and "Financial Services".
    *   **Impact:** Customers engaging in diverse categories (e.g., paying "Utility Bills" *and* buying "Airtime") likely demonstrate higher stickiness and reliability than single-category users. Diversity of spend should be a feature.

3.  **High-Frequency, Low-Value Dominance**
    *   **Finding:** The median transaction value is low compared to the mean.
    *   **Impact:** The "Frequency" component of our RFM model will be a strong driver. A user who transacts small amounts daily is often a better credit risk (more observable behavioral data) than one who transacts once a month.

4.  **Absence of Default Flags Necessitates Unsupervised Grouping**
    *   **Finding:** No `Default` column exists; we only have `FraudResult`.
    *   **Impact:** We cannot train a supervised model immediately. We must first aggregate data to the Customer Level (RFM) and define a "Good/Bad" threshold (e.g., "Good" = Top 25% RFM score) to create our proxy target.

---

## 5. Conclusion and Readiness for Next Phase

**Status: READY for Feature Engineering**

Task 1 and Task 2 have established a solid foundation:
1.  **Business Alignment:** We have defined a clear path to align our Alternative Data scoring with **Basel II** standards by prioritizing interpretability and robust documentation.
2.  **Data Validity:** The **Xente** dataset is clean (no missing values), granular, and rich in behavioral signals suitable for RFM analysis.
3.  **Modeling Strategy:** EDA confirms the need for variable transformation (to handle skew) and rigorous feature selection (to handle correlation).

**Next Steps:**
- Aggregate transaction logs to create a **Customer-Level Table**.
- Calculate **RFM Scores** for every customer.
- Define the **Proxy Target** (Good vs. Bad) based on these scores.
- Begin **Weight of Evidence (WoE)** binning to prepare for the Logistic Regression Scorecard.

The project is on track to deliver a high-quality, defensible credit scoring model.
