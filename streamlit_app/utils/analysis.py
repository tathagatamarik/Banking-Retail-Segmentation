"""
Analysis utilities for Customer Segmentation — Retail Banking
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


def load_data(data_dir="data"):
    customers = pd.read_csv(f"{data_dir}/customers.csv")
    transactions = pd.read_csv(f"{data_dir}/transactions.csv")
    products = pd.read_csv(f"{data_dir}/products_held.csv")
    return customers, transactions, products


FEATURE_COLS = [
    "checking_balance", "savings_balance", "money_market_balance", "cd_balance",
    "mortgage_balance", "auto_loan_balance", "personal_loan_balance",
    "credit_card_balance", "credit_utilization", "investment_balance",
    "total_deposits", "total_lending", "total_relationship_value",
    "product_count", "monthly_transactions", "avg_transaction_amount",
    "recency_days", "branch_visits_monthly", "call_center_monthly",
    "digital_score", "monthly_logins", "mobile_pct", "features_adopted",
    "monthly_revenue", "annual_revenue", "age", "tenure_years", "income",
]


def compute_rfm(df):
    """Compute RFM scores."""
    rfm = df[["customer_id", "recency_days", "monthly_transactions", "annual_revenue"]].copy()
    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]

    def rfm_label(row):
        if row["R_score"] >= 4 and row["F_score"] >= 4 and row["M_score"] >= 4:
            return "Champions"
        elif row["R_score"] >= 3 and row["F_score"] >= 3 and row["M_score"] >= 3:
            return "Loyal"
        elif row["R_score"] >= 4 and row["F_score"] <= 2:
            return "New Customers"
        elif row["R_score"] <= 2 and row["F_score"] >= 3:
            return "At Risk"
        elif row["R_score"] <= 2 and row["F_score"] <= 2 and row["M_score"] <= 2:
            return "Hibernating"
        elif row["M_score"] >= 4:
            return "High Value"
        else:
            return "Needs Attention"
    rfm["rfm_segment"] = rfm.apply(rfm_label, axis=1)
    return rfm


def run_pca(df, features=None, n_components=10):
    """Run PCA on customer features."""
    if features is None:
        features = FEATURE_COLS
    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, len(features)))
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(var_explained)
    loadings = pd.DataFrame(
        pca.components_.T, index=features,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    return X_pca, X_scaled, pca, var_explained, cumulative_var, loadings, scaler


def run_kmeans(X, k_range=range(3, 11), selected_k=7):
    """Run K-Means with elbow and silhouette analysis."""
    inertias, sil_scores = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias[k] = km.inertia_
        sil_scores[k] = silhouette_score(X, labels)
    # Final model
    km_final = KMeans(n_clusters=selected_k, random_state=42, n_init=10, max_iter=300)
    labels_final = km_final.fit_predict(X)
    return labels_final, km_final, inertias, sil_scores


def run_gmm(X, k_range=range(3, 11), selected_k=7):
    """Run Gaussian Mixture Model with BIC analysis."""
    bics, aics = {}, {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full", max_iter=200)
        gmm.fit(X)
        bics[k] = gmm.bic(X)
        aics[k] = gmm.aic(X)
    # Final model
    gmm_final = GaussianMixture(n_components=selected_k, random_state=42, covariance_type="full", max_iter=200)
    gmm_final.fit(X)
    labels_final = gmm_final.predict(X)
    probabilities = gmm_final.predict_proba(X)
    return labels_final, probabilities, gmm_final, bics, aics


def compute_clv(df, discount_rate=0.08, horizon_years=5):
    """Compute simplified Customer Lifetime Value."""
    clv = df[["customer_id", "annual_revenue", "churn_probability", "tenure_years",
              "product_count", "age", "digital_score"]].copy()

    # Growth rate based on age/lifecycle
    clv["growth_rate"] = np.where(clv["age"] < 35, 0.10,
                          np.where(clv["age"] < 50, 0.05,
                          np.where(clv["age"] < 65, 0.02, -0.02)))

    # Retention rate
    clv["retention_rate"] = 1 - clv["churn_probability"]

    # Calculate CLV
    clv_values = []
    for _, row in clv.iterrows():
        total_clv = 0
        for t in range(1, horizon_years + 1):
            revenue_t = row["annual_revenue"] * (1 + row["growth_rate"]) ** t
            survival_t = row["retention_rate"] ** t
            margin = 0.35  # Net margin
            total_clv += (revenue_t * margin * survival_t) / (1 + discount_rate) ** t
        clv_values.append(round(total_clv, 2))

    clv["clv_5year"] = clv_values

    # CLV tiers
    clv["clv_tier"] = pd.qcut(clv["clv_5year"], 4, labels=["Bronze", "Silver", "Gold", "Platinum"])

    return clv


def compute_cohort_metrics(df, transactions_df):
    """Compute cohort analysis metrics."""
    # Create quarterly cohorts based on tenure
    df = df.copy()
    tenure_months = (df["tenure_years"] * 12).astype(int)
    df["cohort_quarter"] = pd.cut(tenure_months, bins=[0, 3, 6, 12, 24, 48, 120, 500],
                                   labels=["0-3m", "3-6m", "6-12m", "1-2y", "2-4y", "4-10y", "10y+"])

    cohort_metrics = df.groupby("cohort_quarter", observed=True).agg(
        count=("customer_id", "count"),
        avg_products=("product_count", "mean"),
        avg_revenue=("annual_revenue", "mean"),
        avg_digital=("digital_score", "mean"),
        avg_churn=("churn_probability", "mean"),
        avg_relationship=("total_relationship_value", "mean"),
    ).round(2)

    return cohort_metrics, df["cohort_quarter"]
