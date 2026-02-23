# Customer Segmentation — Retail Banking

## Relationship Deepening & Product Cross-Sell Optimization

A comprehensive Customer Segmentation project demonstrating how six advanced analytical techniques can transform a retail bank's approach to customer relationship management. Includes a detailed case study document and an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
├── case_study/
│   └── customer_segmentation.pdf      # Comprehensive case study (24 pages)
├── streamlit_app/
│   ├── app.py                         # Main Streamlit application
│   ├── generate_data.py               # Synthetic data generator
│   ├── requirements.txt               # Python dependencies
│   ├── .streamlit/
│   │   └── config.toml                # Dark theme configuration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── analysis.py                # Core analysis functions
│   └── data/                          # Generated synthetic data
│       ├── customers.csv
│       ├── transactions.csv
│       └── products_held.csv
└── README.md
```

##  Quick Start

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 📊 Dashboard Pages

| Page | Algorithm | Key Visualizations |
|------|-----------|-------------------|
| **Executive Dashboard** | Overview | KPIs, segment profiles, deposits vs lending scatter, digital vs revenue |
| **RFM Analysis** | Recency-Frequency-Monetary | R/F/M distributions, score heatmap, 3D RFM space, segment cross-tab |
| **K-Means Clustering** | K-Means | 2D/3D cluster plots, elbow method, silhouette analysis, radar profiles |
| **Gaussian Mixture Models** | GMM (EM Algorithm) | Soft assignments, probability heatmap, BIC/AIC curves, K-Means vs GMM |
| **PCA Visualization** | Principal Component Analysis | Variance explained, loading heatmaps, feature importance, projections |
| **CLV & Cohort Analysis** | CLV + Cohort | CLV distributions, tier economics, age vs CLV, cohort retention curves |

---

## 🔬 Algorithms Implemented

| Algorithm | Purpose | Key Math |
|-----------|---------|----------|
| **RFM Analysis** | Behavioral scoring | Quintile-based R, F, M scoring (1–5 scale) |
| **K-Means** | Hard clustering | Minimize WCSS: Σ‖xᵢ - μₖ‖² with K-Means++ init |
| **GMM** | Soft/probabilistic clustering | EM algorithm: E-step (responsibilities), M-step (parameter updates) |
| **PCA** | Dimensionality reduction | Eigendecomposition of covariance matrix, variance explained |
| **CLV** | Economic prioritization | DCF model with survival probability and segment growth rates |
| **Cohort Analysis** | Lifecycle tracking | Retention matrices, product adoption curves, revenue maturation |

---

## 📋 Case Study Highlights

- **Client**: Top-25 U.S. retail bank — 840 branches, 6.2M customers, $185B assets
- **Challenge**: 2.1 products/customer (benchmark: 3.2–3.8), 14% attrition, 34% product attachment
- **Results**:
  - 58% product attachment improvement (34% → 53.7%)
  - 39% attrition reduction (14% → 8.5%)
  - 31% revenue per customer increase
  - 280% cross-sell conversion improvement
  - **$127M incremental annual revenue** (8-month payback)

---

## 🔧 Configuration

Sidebar controls:
- **Number of Clusters (K)**: 3–10 for K-Means and GMM
- **PCA Components**: 3–15 for dimensionality reduction

---

## 📝 Synthetic Data

- **8,000 customers** across 7 behavioral segments
- **28 features**: balances, transactions, digital engagement, credit, demographics
- **7 ground-truth segments**: Young Professionals, Established Families, Wealth Builders, Digital Enthusiasts, Mature Traditionalists, Small Business Owners, At-Risk/Disengaging
- **182K+ monthly transaction records** spanning 24 months
- **34K+ product holding records** across 10 product categories

---

## 📄 License

Provided for demonstration and educational purposes.
