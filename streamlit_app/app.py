"""
Customer Segmentation — Retail Banking
Comprehensive Interactive Demo Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.analysis import (load_data, FEATURE_COLS, compute_rfm, run_pca, run_kmeans, run_gmm, compute_clv, compute_cohort_metrics)

st.set_page_config(page_title="Customer Segmentation — Retail Banking", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .kpi-card { background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%); border: 1px solid rgba(38, 166, 154, 0.2); border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .kpi-card:hover { border-color: rgba(38, 166, 154, 0.5); box-shadow: 0 4px 20px rgba(38, 166, 154, 0.1); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #26A69A; margin: 0.3rem 0; font-family: 'Inter', sans-serif; }
    .kpi-label { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
    .kpi-delta { font-size: 0.8rem; color: #66BB6A; font-weight: 600; }
    .section-header { font-size: 1.4rem; font-weight: 600; color: #E0E0E0; border-bottom: 2px solid #26A69A; padding-bottom: 0.5rem; margin-bottom: 1rem; }
    .highlight-box { background: rgba(38, 166, 154, 0.08); border-left: 4px solid #26A69A; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
    .algo-card { background: linear-gradient(135deg, #1a2332, #1f2b3d); border: 1px solid rgba(38, 166, 154, 0.15); border-radius: 10px; padding: 1.2rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

PL = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=12), margin=dict(l=40, r=40, t=50, b=40))
C8 = ["#26A69A", "#EF5350", "#42A5F5", "#FFA726", "#AB47BC", "#66BB6A", "#FF7043", "#78909C"]

def kpi_card(label, value, delta=None):
    d = f'<div class="kpi-delta">▲ {delta}</div>' if delta else ""
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{d}</div>'

@st.cache_data
def get_data():
    dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(os.path.join(dd, "customers.csv")):
        from generate_data import generate_data
        generate_data(n_customers=8000, output_dir=dd)
    return load_data(dd)

@st.cache_data
def cached_pca(_df, n): return run_pca(_df, n_components=n)
@st.cache_data
def cached_km(_X, k): return run_kmeans(_X, selected_k=k)
@st.cache_data
def cached_gmm(_X, k): return run_gmm(_X, selected_k=k)

customers, transactions, products = get_data()

with st.sidebar:
    st.markdown("## 🏦 Segmentation Dashboard")
    st.markdown("**Retail Banking Analytics**")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Executive Dashboard", "💎 RFM Analysis", "🎯 K-Means Clustering", "🔮 Gaussian Mixture Models", "📐 PCA Visualization", "💰 CLV & Cohort Analysis"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("##### ⚙️ Settings")
    n_clusters = st.slider("Number of Clusters (K)", 3, 10, 7)
    n_pca = st.slider("PCA Components", 3, 15, 10)
    st.markdown("---")
    st.caption("Synthetic data — 8,000 customers")

X_pca, X_scaled, pca_model, var_exp, cum_var, loadings, scaler = cached_pca(customers, n_pca)

# ===== EXECUTIVE DASHBOARD =====
if page == "📊 Executive Dashboard":
    st.markdown("# 📊 Executive Dashboard")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Retail Banking — Customer Segmentation Overview</p>', unsafe_allow_html=True)
    cols = st.columns(6)
    for col, (l, v, d) in zip(cols, [
        ("Total Customers", f"{len(customers):,}", None), ("Avg Products", f"{customers['product_count'].mean():.1f}", "+58%"),
        ("Avg Revenue", f"${customers['annual_revenue'].mean():,.0f}", "+31%"), ("Digital Score", f"{customers['digital_score'].mean():.0f}/100", "+44%"),
        ("Churn Risk", f"{customers['churn_probability'].mean():.1%}", "-39%"), ("Avg Relationship", f"${customers['total_relationship_value'].mean():,.0f}", None)]):
        with col: st.markdown(kpi_card(l, v, d), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sc = customers["segment_true"].value_counts().reset_index(); sc.columns = ["Segment", "Count"]
        fig = px.bar(sc, x="Count", y="Segment", orientation="h", color="Segment", color_discrete_sequence=C8)
        fig.update_layout(**PL, title="Customer Segments (Ground Truth)", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(customers, x="product_count", color="segment_true", barmode="stack", color_discrete_sequence=C8, nbins=10)
        fig.update_layout(**PL, title="Product Count by Segment", height=400, xaxis_title="Products Held", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        cust_plot = customers.copy(); cust_plot["size_revenue"] = cust_plot["annual_revenue"].clip(lower=0)
        fig = px.scatter(cust_plot, x="total_deposits", y="total_lending", color="segment_true", size="size_revenue", opacity=0.5, color_discrete_sequence=C8)
        fig.update_layout(**PL, title="Deposits vs Lending (size=Revenue)", height=420, xaxis_title="Total Deposits ($)", yaxis_title="Total Lending ($)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(customers, x="digital_score", y="annual_revenue", color="segment_true", opacity=0.4, color_discrete_sequence=C8)
        fig.update_layout(**PL, title="Digital Score vs Annual Revenue", height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Segment Profile Summary</div>', unsafe_allow_html=True)
    ss = customers.groupby("segment_true").agg(count=("customer_id","count"), avg_age=("age","mean"), avg_income=("income","mean"), avg_products=("product_count","mean"), avg_deposits=("total_deposits","mean"), avg_revenue=("annual_revenue","mean"), avg_digital=("digital_score","mean"), avg_churn=("churn_probability","mean")).round(1)
    ss["avg_income"]=ss["avg_income"].apply(lambda x:f"${x:,.0f}"); ss["avg_deposits"]=ss["avg_deposits"].apply(lambda x:f"${x:,.0f}"); ss["avg_revenue"]=ss["avg_revenue"].apply(lambda x:f"${x:,.0f}"); ss["avg_churn"]=ss["avg_churn"].apply(lambda x:f"{x:.1%}")
    ss.columns=["Count","Avg Age","Avg Income","Avg Products","Avg Deposits","Avg Revenue","Avg Digital","Churn Risk"]
    st.dataframe(ss, use_container_width=True)

# ===== RFM =====
elif page == "💎 RFM Analysis":
    st.markdown("# 💎 RFM Analysis")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Recency, Frequency, Monetary scoring for behavioral segmentation</p>', unsafe_allow_html=True)
    rfm = compute_rfm(customers)
    rm = rfm.merge(customers[["customer_id","segment_true","product_count","digital_score"]], on="customer_id")
    c1,c2,c3 = st.columns(3)
    with c1:
        fig=px.histogram(rfm,x="recency",nbins=40,color_discrete_sequence=["#EF5350"]); fig.update_layout(**PL,title="Recency (Days)",height=280); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.histogram(rfm,x="frequency",nbins=40,color_discrete_sequence=["#26A69A"]); fig.update_layout(**PL,title="Frequency (Monthly Txns)",height=280); st.plotly_chart(fig,use_container_width=True)
    with c3:
        fig=px.histogram(rfm,x="monetary",nbins=40,color_discrete_sequence=["#FFA726"]); fig.update_layout(**PL,title="Monetary (Annual Rev $)",height=280); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section-header">RFM Score Heatmap</div>', unsafe_allow_html=True)
    rc=pd.crosstab(rfm["R_score"],rfm["F_score"])
    fig=px.imshow(rc,color_continuous_scale=["#0E1117","#1a2744","#26A69A","#FFA726"],labels=dict(x="Frequency Score",y="Recency Score",color="Count"),aspect="auto")
    fig.update_layout(**PL,height=400); st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        sc2=rfm["rfm_segment"].value_counts().reset_index(); sc2.columns=["Segment","Count"]
        fig=px.pie(sc2,values="Count",names="Segment",hole=0.4,color_discrete_sequence=C8); fig.update_layout(**PL,title="RFM Segment Distribution",height=400); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.scatter_3d(rm,x="recency",y="frequency",z="monetary",color="rfm_segment",opacity=0.5,color_discrete_sequence=C8)
        fig.update_layout(**PL,title="3D RFM Space",height=450); st.plotly_chart(fig,use_container_width=True)
    with st.expander("📊 RFM Segment vs True Segment"):
        cr=pd.crosstab(rm["rfm_segment"],rm["segment_true"],normalize="index").round(3)*100
        fig=px.imshow(cr,color_continuous_scale=["#0E1117","#26A69A","#EF5350"],labels=dict(color="% of RFM Seg"),aspect="auto")
        fig.update_layout(**PL,title="",height=400); st.plotly_chart(fig,use_container_width=True)

# ===== K-MEANS =====
elif page == "🎯 K-Means Clustering":
    st.markdown("# 🎯 K-Means Clustering")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Unsupervised segment discovery via centroid-based partitioning</p>', unsafe_allow_html=True)
    lkm,km,inertias,sils = cached_km(X_pca, n_clusters)
    ckm = customers.copy(); ckm["cluster"]=lkm
    t1,t2,t3 = st.tabs(["🎯 Clusters","📈 Elbow & Silhouette","🔍 Profiles"])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(x=X_pca[:,0],y=X_pca[:,1],color=lkm.astype(str),opacity=0.5,color_discrete_sequence=C8,labels={"x":f"PC1 ({var_exp[0]:.1%})","y":f"PC2 ({var_exp[1]:.1%})","color":"Cluster"})
            fig.update_layout(**PL,title="K-Means (PC1 vs PC2)",height=480); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.scatter_3d(x=X_pca[:,0],y=X_pca[:,1],z=X_pca[:,2],color=lkm.astype(str),opacity=0.4,color_discrete_sequence=C8,labels={"x":"PC1","y":"PC2","z":"PC3","color":"Cluster"})
            fig.update_layout(**PL,title="3D Cluster View",height=480); st.plotly_chart(fig,use_container_width=True)
        st.markdown(f'<div class="highlight-box">K={n_clusters} — Silhouette: <b>{sils[n_clusters]:.3f}</b></div>', unsafe_allow_html=True)
    with t2:
        c1,c2=st.columns(2)
        with c1:
            ed=pd.DataFrame(list(inertias.items()),columns=["K","WCSS"])
            fig=px.line(ed,x="K",y="WCSS",markers=True,color_discrete_sequence=["#26A69A"]); fig.add_vline(x=n_clusters,line_dash="dash",line_color="#EF5350")
            fig.update_layout(**PL,title="Elbow Method (WCSS)",height=400); st.plotly_chart(fig,use_container_width=True)
        with c2:
            sd=pd.DataFrame(list(sils.items()),columns=["K","Silhouette"])
            fig=px.line(sd,x="K",y="Silhouette",markers=True,color_discrete_sequence=["#FFA726"]); fig.add_vline(x=n_clusters,line_dash="dash",line_color="#EF5350")
            fig.update_layout(**PL,title="Silhouette Score",height=400); st.plotly_chart(fig,use_container_width=True)
    with t3:
        cp=ckm.groupby("cluster").agg(size=("customer_id","count"),avg_age=("age","mean"),avg_income=("income","mean"),avg_products=("product_count","mean"),avg_deposits=("total_deposits","mean"),avg_lending=("total_lending","mean"),avg_revenue=("annual_revenue","mean"),avg_digital=("digital_score","mean"),avg_churn=("churn_probability","mean")).round(1)
        cp["pct"]=(cp["size"]/cp["size"].sum()*100).round(1); st.dataframe(cp,use_container_width=True)
        st.markdown('<div class="section-header">Cluster Radar Profiles</div>', unsafe_allow_html=True)
        rf=["avg_income","avg_products","avg_deposits","avg_revenue","avg_digital"]; rl=["Income","Products","Deposits","Revenue","Digital"]
        rd=cp[rf].copy()
        for col in rf:
            mx=rd[col].max()
            if mx>0: rd[col]=rd[col]/mx
        fig=go.Figure()
        for i,(idx,row) in enumerate(rd.iterrows()):
            vals=row.tolist()+[row.tolist()[0]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=rl+[rl[0]],fill="toself",name=f"Cluster {idx}",line_color=C8[i%len(C8)],opacity=0.6))
        fig.update_layout(**PL,height=450,polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,0.1)"),angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")))
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("🔍 Cluster vs True Segment"):
            cr=pd.crosstab(ckm["cluster"],ckm["segment_true"],normalize="index").round(3)*100
            fig=px.imshow(cr,color_continuous_scale=["#0E1117","#26A69A","#FFA726"],labels=dict(color="% of Cluster"),aspect="auto")
            fig.update_layout(**PL,height=400); st.plotly_chart(fig,use_container_width=True)

# ===== GMM =====
elif page == "🔮 Gaussian Mixture Models":
    st.markdown("# 🔮 Gaussian Mixture Models")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Probabilistic soft-assignment via Expectation-Maximization</p>', unsafe_allow_html=True)
    lg,pg,gm,bics,aics = cached_gmm(X_pca, n_clusters)
    cg=customers.copy(); cg["gmm_cluster"]=lg; cg["max_prob"]=pg.max(axis=1)
    t1,t2,t3=st.tabs(["🔮 Clusters & Probabilities","📊 BIC/AIC","🔄 K-Means vs GMM"])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(x=X_pca[:,0],y=X_pca[:,1],color=lg.astype(str),opacity=0.5,color_discrete_sequence=C8,labels={"x":f"PC1","y":f"PC2","color":"Cluster"})
            fig.update_layout(**PL,title="GMM Clusters",height=450); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.histogram(cg,x="max_prob",nbins=50,color_discrete_sequence=["#26A69A"])
            fig.add_vline(x=0.65,line_dash="dash",line_color="#EF5350",annotation_text="Boundary (0.65)")
            fig.update_layout(**PL,title="Max Assignment Probability",height=450,xaxis_title="Probability"); st.plotly_chart(fig,use_container_width=True)
        bp=(cg["max_prob"]<0.65).mean()
        st.markdown(f'<div class="highlight-box"><b>{bp:.1%}</b> boundary customers (max prob &lt; 0.65) — highest cross-sell targets due to multi-segment needs.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Sample Probability Profiles</div>', unsafe_allow_html=True)
        si=np.random.RandomState(42).choice(len(cg),30,replace=False)
        sp=pd.DataFrame(pg[si],columns=[f"C{i}" for i in range(n_clusters)]); sp.index=cg.iloc[si]["customer_id"].values
        fig=px.imshow(sp.T,color_continuous_scale=["#0E1117","#1a2744","#26A69A","#FFA726"],labels=dict(x="Customer",y="Cluster",color="Prob"),aspect="auto")
        fig.update_layout(**PL,height=280); fig.update_xaxes(tickangle=45,tickfont_size=8); st.plotly_chart(fig,use_container_width=True)
    with t2:
        c1,c2=st.columns(2)
        with c1:
            bd=pd.DataFrame(list(bics.items()),columns=["K","BIC"])
            fig=px.line(bd,x="K",y="BIC",markers=True,color_discrete_sequence=["#26A69A"]); fig.add_vline(x=n_clusters,line_dash="dash",line_color="#EF5350")
            fig.update_layout(**PL,title="BIC",height=380); st.plotly_chart(fig,use_container_width=True)
        with c2:
            ad=pd.DataFrame(list(aics.items()),columns=["K","AIC"])
            fig=px.line(ad,x="K",y="AIC",markers=True,color_discrete_sequence=["#FFA726"]); fig.add_vline(x=n_clusters,line_dash="dash",line_color="#EF5350")
            fig.update_layout(**PL,title="AIC",height=380); st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="highlight-box"><b>BIC</b> = -2·log(L) + p·ln(n) — penalizes complexity. <b>AIC</b> = -2·log(L) + 2p — less penalty. Lower = better.</div>', unsafe_allow_html=True)
    with t3:
        lkc,_,_,_=cached_km(X_pca,n_clusters)
        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(x=X_pca[:,0],y=X_pca[:,1],color=lkc.astype(str),opacity=0.4,color_discrete_sequence=C8,labels={"x":"PC1","y":"PC2","color":"Cluster"})
            fig.update_layout(**PL,title="K-Means (Hard)",height=400); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.scatter(x=X_pca[:,0],y=X_pca[:,1],color=cg["max_prob"],color_continuous_scale=["#EF5350","#FFA726","#26A69A"],opacity=0.5,labels={"x":"PC1","y":"PC2","color":"Certainty"})
            fig.update_layout(**PL,title="GMM (Certainty-colored)",height=400); st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="highlight-box"><b>Key:</b> K-Means forces hard assignment. GMM gives probability vectors — a customer can be 60% "Digital Enthusiast" + 40% "Wealth Builder", enabling hybrid strategies. Red/orange points = boundary customers where GMM adds most value.</div>', unsafe_allow_html=True)

# ===== PCA =====
elif page == "📐 PCA Visualization":
    st.markdown("# 📐 Principal Component Analysis")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Dimensionality reduction: 28 features → interpretable components</p>', unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["📊 Variance","🔗 Loadings","🗺️ Projection"])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            vd=pd.DataFrame({"PC":[f"PC{i+1}" for i in range(len(var_exp))],"Variance":var_exp,"Cumulative":cum_var})
            fig=px.bar(vd,x="PC",y="Variance",color_discrete_sequence=["#26A69A"])
            fig.add_trace(go.Scatter(x=vd["PC"],y=vd["Cumulative"],mode="lines+markers",name="Cumulative",line=dict(color="#FFA726",width=2.5)))
            fig.add_hline(y=0.90,line_dash="dash",line_color="#EF5350",annotation_text="90%")
            fig.update_layout(**PL,title="Variance Explained",height=420,yaxis_title="Proportion"); st.plotly_chart(fig,use_container_width=True)
        with c2:
            n90=np.argmax(cum_var>=0.90)+1 if any(cum_var>=0.90) else len(cum_var)
            st.markdown(f'<div class="algo-card"><h4 style="color:#26A69A;margin-top:0;">PCA Summary</h4><p><b>Original:</b> {len(FEATURE_COLS)} features</p><p><b>90% variance:</b> {n90} components</p><p><b>Compression:</b> {len(FEATURE_COLS)/n90:.1f}x</p><p><b>PC1:</b> {var_exp[0]:.1%} (Relationship Depth)</p><p><b>PC2:</b> {var_exp[1]:.1%} (Digital Engagement)</p></div>', unsafe_allow_html=True)
    with t2:
        pc_sel=st.selectbox("Component",[f"PC{i+1}" for i in range(min(5,len(var_exp)))])
        tp=loadings[pc_sel].nlargest(8).reset_index(); tn=loadings[pc_sel].nsmallest(8).reset_index()
        tp.columns=["Feature","Loading"]; tn.columns=["Feature","Loading"]
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(tp,x="Loading",y="Feature",orientation="h",color_discrete_sequence=["#26A69A"]); fig.update_layout(**PL,title=f"{pc_sel} Positive",height=350,yaxis=dict(autorange="reversed")); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.bar(tn,x="Loading",y="Feature",orientation="h",color_discrete_sequence=["#EF5350"]); fig.update_layout(**PL,title=f"{pc_sel} Negative",height=350,yaxis=dict(autorange="reversed")); st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="section-header">Full Loading Heatmap</div>', unsafe_allow_html=True)
        ld=loadings.iloc[:,:5]
        fig=px.imshow(ld,color_continuous_scale=["#EF5350","#0E1117","#26A69A"],labels=dict(color="Loading"),aspect="auto",zmin=-0.5,zmax=0.5)
        fig.update_layout(**PL,height=600); fig.update_yaxes(tickfont_size=9); st.plotly_chart(fig,use_container_width=True)
    with t3:
        cb=st.selectbox("Color by",["segment_true","product_count","digital_score","annual_revenue","churn_probability"])
        fig=px.scatter(x=X_pca[:,0],y=X_pca[:,1],color=customers[cb],color_discrete_sequence=C8 if customers[cb].dtype=="object" else None,color_continuous_scale=["#0E1117","#26A69A","#FFA726"] if customers[cb].dtype!="object" else None,opacity=0.4,labels={"x":f"PC1 ({var_exp[0]:.1%})","y":f"PC2 ({var_exp[1]:.1%})","color":cb})
        fig.update_layout(**PL,title=f"PCA colored by {cb}",height=550); st.plotly_chart(fig,use_container_width=True)

# ===== CLV & COHORT =====
elif page == "💰 CLV & Cohort Analysis":
    st.markdown("# 💰 CLV & Cohort Analysis")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Customer Lifetime Value & cohort lifecycle tracking</p>', unsafe_allow_html=True)
    t1,t2=st.tabs(["💰 CLV","📅 Cohort"])
    with t1:
        clv=compute_clv(customers); cm=clv.merge(customers[["customer_id","segment_true","total_relationship_value"]],on="customer_id")
        cols=st.columns(4)
        for col,(l,v) in zip(cols,[("Avg 5Y CLV",f"${clv['clv_5year'].mean():,.0f}"),("Median CLV",f"${clv['clv_5year'].median():,.0f}"),("Top 10% CLV",f"${clv['clv_5year'].quantile(0.9):,.0f}"),("Total CLV Pool",f"${clv['clv_5year'].sum()/1e6:,.0f}M")]):
            with col: st.markdown(kpi_card(l,v),unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(clv,x="clv_5year",nbins=50,color="clv_tier",color_discrete_sequence=["#78909C","#42A5F5","#FFA726","#26A69A"])
            fig.update_layout(**PL,title="CLV Distribution by Tier",height=400,xaxis_title="5-Year CLV ($)"); st.plotly_chart(fig,use_container_width=True)
        with c2:
            ts=clv.groupby("clv_tier",observed=True).agg(count=("customer_id","count"),avg_clv=("clv_5year","mean"),avg_rev=("annual_revenue","mean"),avg_ch=("churn_probability","mean")).round(0)
            ts["pct"]=(ts["count"]/ts["count"].sum()*100).round(1); ts["pct_val"]=(ts["count"]*ts["avg_clv"]); ts["pct_val"]=(ts["pct_val"]/ts["pct_val"].sum()*100).round(1)
            ts=ts[["count","pct","avg_clv","pct_val","avg_rev","avg_ch"]]; ts.columns=["Customers","%Cust","Avg CLV ($)","%CLV","Avg Rev ($)","Churn"]
            ts["Churn"]=ts["Churn"].apply(lambda x:f"{x:.1%}"); st.dataframe(ts,use_container_width=True)
        st.markdown('<div class="section-header">CLV by Segment</div>',unsafe_allow_html=True)
        fig=px.box(cm,x="segment_true",y="clv_5year",color="segment_true",color_discrete_sequence=C8)
        fig.update_layout(**PL,title="",height=400,showlegend=False,xaxis_title="Segment",yaxis_title="5-Year CLV ($)"); st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(cm,x="annual_revenue",y="clv_5year",color="segment_true",opacity=0.4,color_discrete_sequence=C8)
            fig.update_layout(**PL,title="Current Revenue vs CLV",height=420); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.scatter(cm,x="age",y="clv_5year",color="segment_true",opacity=0.4,color_discrete_sequence=C8)
            fig.update_layout(**PL,title="Age vs CLV",height=420); st.plotly_chart(fig,use_container_width=True)
    with t2:
        cm2,cl2=compute_cohort_metrics(customers,transactions); st.dataframe(cm2,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(cm2.reset_index(),x="cohort_quarter",y="avg_products",color_discrete_sequence=["#26A69A"])
            fig.update_layout(**PL,title="Products by Tenure Cohort",height=350); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.bar(cm2.reset_index(),x="cohort_quarter",y="avg_revenue",color_discrete_sequence=["#FFA726"])
            fig.update_layout(**PL,title="Revenue by Tenure Cohort",height=350); st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(cm2.reset_index(),x="cohort_quarter",y="avg_churn",color_discrete_sequence=["#EF5350"])
            fig.update_layout(**PL,title="Churn Risk by Cohort",height=350); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.bar(cm2.reset_index(),x="cohort_quarter",y="avg_digital",color_discrete_sequence=["#42A5F5"])
            fig.update_layout(**PL,title="Digital Score by Cohort",height=350); st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="highlight-box"><b>Cohort insight:</b> Newer cohorts show higher churn but higher digital scores. The "golden 90 days" — product adoption in first 3 months predicts long-term retention. Mature cohorts (10y+) have highest relationship value but lowest digital engagement.</div>', unsafe_allow_html=True)
