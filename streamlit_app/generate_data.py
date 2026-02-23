"""
Synthetic Retail Banking Customer Data Generator
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

SEGMENTS = {
    "Young Professionals": {
        "proportion": 0.18, "age_range": (23, 34), "tenure_range": (0.5, 4),
        "checking_bal": (2000, 12000), "savings_bal": (500, 8000),
        "has_mortgage": 0.08, "has_credit_card": 0.82, "has_auto_loan": 0.30,
        "has_investment": 0.12, "has_personal_loan": 0.15,
        "digital_score": (70, 98), "monthly_txns": (25, 80),
        "credit_util": (0.25, 0.65), "income_range": (45000, 90000),
        "churn_prob": 0.16, "growth_rate": 0.12,
    },
    "Established Families": {
        "proportion": 0.22, "age_range": (32, 50), "tenure_range": (3, 12),
        "checking_bal": (5000, 25000), "savings_bal": (3000, 30000),
        "has_mortgage": 0.65, "has_credit_card": 0.90, "has_auto_loan": 0.45,
        "has_investment": 0.35, "has_personal_loan": 0.10,
        "digital_score": (45, 85), "monthly_txns": (30, 100),
        "credit_util": (0.20, 0.50), "income_range": (70000, 160000),
        "churn_prob": 0.06, "growth_rate": 0.06,
    },
    "Wealth Builders": {
        "proportion": 0.12, "age_range": (38, 58), "tenure_range": (5, 20),
        "checking_bal": (15000, 80000), "savings_bal": (20000, 150000),
        "has_mortgage": 0.55, "has_credit_card": 0.95, "has_auto_loan": 0.25,
        "has_investment": 0.80, "has_personal_loan": 0.05,
        "digital_score": (50, 88), "monthly_txns": (20, 60),
        "credit_util": (0.05, 0.25), "income_range": (120000, 350000),
        "churn_prob": 0.04, "growth_rate": 0.08,
    },
    "Digital Enthusiasts": {
        "proportion": 0.15, "age_range": (22, 40), "tenure_range": (0.5, 6),
        "checking_bal": (3000, 18000), "savings_bal": (1000, 15000),
        "has_mortgage": 0.15, "has_credit_card": 0.88, "has_auto_loan": 0.20,
        "has_investment": 0.25, "has_personal_loan": 0.18,
        "digital_score": (85, 100), "monthly_txns": (40, 120),
        "credit_util": (0.20, 0.55), "income_range": (55000, 130000),
        "churn_prob": 0.12, "growth_rate": 0.10,
    },
    "Mature Traditionalists": {
        "proportion": 0.16, "age_range": (55, 78), "tenure_range": (10, 35),
        "checking_bal": (8000, 45000), "savings_bal": (15000, 120000),
        "has_mortgage": 0.20, "has_credit_card": 0.75, "has_auto_loan": 0.10,
        "has_investment": 0.55, "has_personal_loan": 0.03,
        "digital_score": (5, 40), "monthly_txns": (10, 35),
        "credit_util": (0.02, 0.15), "income_range": (40000, 95000),
        "churn_prob": 0.05, "growth_rate": -0.02,
    },
    "Small Business Owners": {
        "proportion": 0.08, "age_range": (30, 58), "tenure_range": (2, 15),
        "checking_bal": (10000, 60000), "savings_bal": (5000, 40000),
        "has_mortgage": 0.50, "has_credit_card": 0.92, "has_auto_loan": 0.30,
        "has_investment": 0.40, "has_personal_loan": 0.25,
        "digital_score": (40, 80), "monthly_txns": (50, 150),
        "credit_util": (0.30, 0.70), "income_range": (60000, 250000),
        "churn_prob": 0.10, "growth_rate": 0.09,
    },
    "At-Risk / Disengaging": {
        "proportion": 0.09, "age_range": (25, 65), "tenure_range": (1, 10),
        "checking_bal": (200, 3000), "savings_bal": (0, 1500),
        "has_mortgage": 0.05, "has_credit_card": 0.55, "has_auto_loan": 0.12,
        "has_investment": 0.03, "has_personal_loan": 0.08,
        "digital_score": (5, 35), "monthly_txns": (2, 15),
        "credit_util": (0.50, 0.95), "income_range": (25000, 55000),
        "churn_prob": 0.35, "growth_rate": -0.10,
    },
}


def generate_data(n_customers=8000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    seg_names = list(SEGMENTS.keys())
    seg_probs = [SEGMENTS[s]["proportion"] for s in seg_names]
    customers, transactions, products_held = [], [], []

    for cid in range(1, n_customers + 1):
        seg = np.random.choice(seg_names, p=seg_probs)
        s = SEGMENTS[seg]
        age = int(np.random.uniform(*s["age_range"]))
        tenure_years = round(np.random.uniform(*s["tenure_range"]), 1)
        gender = np.random.choice(["M", "F"], p=[0.48, 0.52])
        income = int(np.random.uniform(*s["income_range"]))
        region = np.random.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"],
                                   p=[0.25, 0.20, 0.20, 0.15, 0.20])
        checking_bal = round(np.random.uniform(*s["checking_bal"]), 2)
        savings_bal = round(np.random.uniform(*s["savings_bal"]), 2)
        mm_bal = round(np.random.uniform(0, savings_bal * 0.5), 2) if np.random.random() < 0.25 else 0.0
        cd_bal = round(np.random.uniform(5000, 50000), 2) if np.random.random() < (0.15 if age > 45 else 0.05) else 0.0
        mortgage_bal = round(np.random.uniform(80000, 450000), 2) if np.random.random() < s["has_mortgage"] else 0.0
        heloc_bal = round(np.random.uniform(10000, 80000), 2) if mortgage_bal > 0 and np.random.random() < 0.20 else 0.0
        auto_bal = round(np.random.uniform(5000, 40000), 2) if np.random.random() < s["has_auto_loan"] else 0.0
        personal_bal = round(np.random.uniform(2000, 25000), 2) if np.random.random() < s["has_personal_loan"] else 0.0
        has_cc = np.random.random() < s["has_credit_card"]
        cc_limit = round(np.random.uniform(3000, 30000), 2) if has_cc else 0.0
        credit_util = round(np.random.uniform(*s["credit_util"]), 3) if has_cc else 0.0
        cc_balance = round(cc_limit * credit_util, 2) if has_cc else 0.0
        has_invest = np.random.random() < s["has_investment"]
        invest_bal = round(np.random.uniform(5000, 500000), 2) if has_invest else 0.0
        digital_score = round(np.random.uniform(*s["digital_score"]), 1)
        monthly_logins = max(0, int(np.random.normal(digital_score * 0.3, 5)))
        mobile_pct = min(100, max(0, round(digital_score * 0.9 + np.random.normal(0, 10), 1)))
        features_adopted = min(12, max(0, int(digital_score * 0.12 + np.random.normal(0, 1))))
        monthly_txns = int(np.random.uniform(*s["monthly_txns"]))
        avg_txn_amount = round(income / (12 * max(monthly_txns, 1)) * np.random.uniform(0.3, 0.8), 2)
        branch_visits_monthly = max(0, int(np.random.normal(max(1, (100 - digital_score) * 0.06), 1.5)))
        call_center_monthly = max(0, int(np.random.normal(1.5, 1)))

        product_count = 1
        prod_list = ["Checking"]
        if savings_bal > 0: product_count += 1; prod_list.append("Savings")
        if mm_bal > 0: product_count += 1; prod_list.append("Money Market")
        if cd_bal > 0: product_count += 1; prod_list.append("CD")
        if mortgage_bal > 0: product_count += 1; prod_list.append("Mortgage")
        if heloc_bal > 0: product_count += 1; prod_list.append("HELOC")
        if auto_bal > 0: product_count += 1; prod_list.append("Auto Loan")
        if personal_bal > 0: product_count += 1; prod_list.append("Personal Loan")
        if has_cc: product_count += 1; prod_list.append("Credit Card")
        if has_invest: product_count += 1; prod_list.append("Investment Account")

        total_deposits = checking_bal + savings_bal + mm_bal + cd_bal
        total_lending = mortgage_bal + heloc_bal + auto_bal + personal_bal + cc_balance
        total_relationship = total_deposits + total_lending + invest_bal

        deposit_nii = total_deposits * 0.025 / 12
        lending_nii = (mortgage_bal * 0.012 + heloc_bal * 0.025 + auto_bal * 0.030 + personal_bal * 0.050 + cc_balance * 0.08) / 12
        fee_income = (12 if has_cc else 0) + branch_visits_monthly * 0.5 + (15 if invest_bal > 0 else 0)
        service_cost = branch_visits_monthly * 4.20 + call_center_monthly * 2.80 + monthly_logins * 0.12
        monthly_revenue = round(deposit_nii + lending_nii + fee_income - service_cost, 2)
        annual_revenue = round(monthly_revenue * 12, 2)

        churn_base = s["churn_prob"]
        churn_adj = churn_base * (1.5 if product_count == 1 else 1.0) * (0.7 if digital_score > 70 else 1.0)
        churn_prob = min(0.50, max(0.01, churn_adj + np.random.normal(0, 0.03)))
        recency_days = max(0, int(np.random.exponential(15 if seg != "At-Risk / Disengaging" else 60)))

        customers.append({
            "customer_id": f"B{cid:06d}", "segment_true": seg,
            "age": age, "gender": gender, "income": income, "region": region,
            "tenure_years": tenure_years,
            "checking_balance": checking_bal, "savings_balance": savings_bal,
            "money_market_balance": mm_bal, "cd_balance": cd_bal,
            "mortgage_balance": mortgage_bal, "heloc_balance": heloc_bal,
            "auto_loan_balance": auto_bal, "personal_loan_balance": personal_bal,
            "credit_card_balance": cc_balance, "credit_card_limit": cc_limit,
            "credit_utilization": credit_util, "investment_balance": invest_bal,
            "total_deposits": round(total_deposits, 2),
            "total_lending": round(total_lending, 2),
            "total_relationship_value": round(total_relationship, 2),
            "product_count": product_count,
            "monthly_transactions": monthly_txns, "avg_transaction_amount": avg_txn_amount,
            "recency_days": recency_days,
            "branch_visits_monthly": branch_visits_monthly,
            "call_center_monthly": call_center_monthly,
            "digital_score": digital_score, "monthly_logins": monthly_logins,
            "mobile_pct": mobile_pct, "features_adopted": features_adopted,
            "monthly_revenue": monthly_revenue, "annual_revenue": annual_revenue,
            "churn_probability": round(churn_prob, 4),
            "life_stage": np.random.choice(
                ["Starting Out", "Building", "Peak Earning", "Pre-Retirement", "Retired"],
                p=[0.50, 0.35, 0.10, 0.03, 0.02] if seg == "Young Professionals"
                else [0.15, 0.25, 0.30, 0.15, 0.15]),
        })

        for prod in prod_list:
            products_held.append({
                "customer_id": f"B{cid:06d}", "product": prod,
                "open_date": (datetime(2026, 2, 1) - timedelta(days=int(tenure_years * 365 * np.random.uniform(0.3, 1.0)))).strftime("%Y-%m-%d"),
                "status": "Active",
            })

    # Monthly transaction history
    for cust in customers:
        months = min(24, int(cust["tenure_years"] * 12))
        for m in range(max(1, months)):
            txn_date = datetime(2026, 2, 1) - timedelta(days=30 * m)
            n_txns = max(1, int(cust["monthly_transactions"] + np.random.normal(0, 5)))
            total_amount = round(n_txns * cust["avg_transaction_amount"] * np.random.uniform(0.7, 1.3), 2)
            transactions.append({
                "customer_id": cust["customer_id"],
                "month": txn_date.strftime("%Y-%m"),
                "transaction_count": n_txns,
                "total_amount": total_amount,
                "deposits": round(total_amount * np.random.uniform(0.4, 0.7), 2),
                "withdrawals": round(total_amount * np.random.uniform(0.3, 0.6), 2),
                "digital_transactions": int(n_txns * cust["digital_score"] / 100),
            })

    df_c = pd.DataFrame(customers)
    df_t = pd.DataFrame(transactions)
    df_p = pd.DataFrame(products_held)
    df_c.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    df_t.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    df_p.to_csv(os.path.join(output_dir, "products_held.csv"), index=False)
    print(f"Generated {len(df_c)} customers, {len(df_t)} transaction records, {len(df_p)} product holdings")
    return df_c, df_t, df_p

if __name__ == "__main__":
    generate_data(n_customers=8000, output_dir="data")
