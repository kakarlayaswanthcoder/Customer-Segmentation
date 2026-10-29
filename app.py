import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# =====================================================
# LOAD MODEL AND DATA
# =====================================================

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("Customer_Segmentation_Final.csv")


# =====================================================
# CUSTOMER SEGMENT NAMES
# =====================================================

segment_names = {
    0: "High Income - Medium Spending",
    1: "Moderate Income - High Spending",
    2: "High Income - Low Spending",
    3: "High Income - High Spending",
    4: "Moderate Income - Low Spending"
}


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
}

.block-container {
    max-width: 1200px;
    padding-top: 25px;
}

/* Main Header */

.hero {
    background: linear-gradient(135deg, #312e81, #4f46e5, #7c3aed);
    padding: 38px;
    border-radius: 22px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(79,70,229,0.20);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 8px;
    opacity: 0.9;
}


/* Result Box */

.result-box {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #6ee7b7;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 23px;
    font-weight: 800;
    color: #065f46;
}

.result-segment {
    font-size: 27px;
    font-weight: 800;
    color: #047857;
    margin-top: 8px;
}


/* Buttons */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 17px;
    font-weight: 700;
}


/* Footer */

.footer {
    text-align: center;
    color: #6b7280;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
👥 Customer Segmentation System
</div>

<div class="hero-subtitle">
Machine Learning Based Customer Analysis using K-Means Clustering
</div>

<div class="hero-subtitle">
Analyze existing customers and predict segments for new customers
</div>

</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🆕 New Customer",
        "👥 Existing Customers",
        "📉 Elbow Method",
        "📊 Cluster Analysis",
        "ℹ️ About Project"
    ]
)


# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.markdown("## 🏠 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Customers",
            "2,000"
        )

    with col2:
        st.metric(
            "🤖 Algorithm",
            "K-Means"
        )

    with col3:
        st.metric(
            "👥 Clusters",
            "5"
        )

    with col4:
        st.metric(
            "📈 Features",
            "5"
        )

    st.markdown("### 🎯 What Does This System Do?")

    st.write(
        """
        This system groups customers into different segments based on
        their characteristics and behaviour using K-Means Clustering.
        """
    )

    st.markdown("### 🔄 Project Workflow")

    w1, w2, w3, w4, w5 = st.columns(5)

    with w1:
        st.info(
            "① **Data**\n\n"
            "Customer Dataset"
        )

    with w2:
        st.info(
            "② **Preprocessing**\n\n"
            "Clean & Scale"
        )

    with w3:
        st.info(
            "③ **Elbow Method**\n\n"
            "Select K = 5"
        )

    with w4:
        st.info(
            "④ **K-Means**\n\n"
            "Create Clusters"
        )

    with w5:
        st.info(
            "⑤ **Prediction**\n\n"
            "Customer Segment"
        )

    st.markdown("### 💼 Business Use")

    st.success(
        "Businesses can understand different customer groups and "
        "design suitable marketing strategies, offers and recommendations."
    )


# =====================================================
# NEW CUSTOMER PAGE
# =====================================================

elif page == "🆕 New Customer":

    st.markdown("## 🆕 New Customer Prediction")

    st.caption(
        "Enter a new customer's details to predict their customer segment."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### 👤 Personal Details")

        age = st.number_input(
            "🎂 Age",
            min_value=1,
            max_value=100,
            value=25
        )

        family_size = st.number_input(
            "👨‍👩‍👧 Family Size",
            min_value=1,
            max_value=20,
            value=4
        )

        work_experience = st.number_input(
            "💼 Work Experience (Years)",
            min_value=0,
            max_value=60,
            value=3
        )

    with right:

        st.markdown("### 💰 Financial Details")

        income = st.number_input(
            "💵 Annual Income ($)",
            min_value=0,
            value=40000,
            step=1000
        )

        spending_score = st.slider(
            "🛍️ Spending Score",
            min_value=1,
            max_value=100,
            value=70
        )

    st.markdown("---")

    if st.button("🔮 PREDICT CUSTOMER SEGMENT"):

        new_customer = np.array([
            [
                age,
                income,
                spending_score,
                work_experience,
                family_size
            ]
        ])

        # Scale the new customer
        new_customer_scaled = scaler.transform(
            new_customer
        )

        # Predict cluster
        predicted_cluster = kmeans.predict(
            new_customer_scaled
        )[0]

        predicted_segment = segment_names[
            predicted_cluster
        ]

        # Result
        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
            ✅ Prediction Completed
            </div>

            <div style="margin-top:15px;">
            Customer Cluster
            </div>

            <div class="result-segment">
            Cluster {predicted_cluster}
            </div>

            <div style="margin-top:15px;">
            Customer Segment
            </div>

            <div class="result-segment">
            {predicted_segment}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📋 Customer Details")

        r1, r2, r3, r4, r5 = st.columns(5)

        with r1:
            st.metric(
                "Age",
                age
            )

        with r2:
            st.metric(
                "Income",
                f"${income:,.0f}"
            )

        with r3:
            st.metric(
                "Spending",
                spending_score
            )

        with r4:
            st.metric(
                "Experience",
                f"{work_experience} yrs"
            )

        with r5:
            st.metric(
                "Family Size",
                family_size
            )


# =====================================================
# EXISTING CUSTOMERS PAGE
# =====================================================

elif page == "👥 Existing Customers":

    st.markdown("## 👥 Existing Customers")

    st.caption(
        "Explore customers already present in the dataset."
    )

    search_id = st.number_input(
        "🔎 Search Customer ID",
        min_value=1,
        max_value=int(df["CustomerID"].max()),
        value=1
    )

    customer = df[
        df["CustomerID"] == search_id
    ]

    if not customer.empty:

        st.markdown("### 👤 Customer Information")

        st.dataframe(
            customer,
            use_container_width=True,
            hide_index=True
        )

        cluster = int(
            customer["Cluster"].iloc[0]
        )

        st.success(
            f"Customer Segment: "
            f"**{segment_names[cluster]}**"
        )

    else:

        st.warning(
            "Customer ID not found."
        )

    st.markdown("### 📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# ELBOW METHOD PAGE
# =====================================================

elif page == "📉 Elbow Method":

    st.markdown("## 📉 Elbow Method")

    st.write(
        """
        The Elbow Method is used to determine the optimal number
        of clusters for K-Means clustering.
        """
    )

    # Features used in the project

    features = [
        "Age",
        "Annual Income ($)",
        "Spending Score (1-100)",
        "Work Experience",
        "Family Size"
    ]

    X = df[features]

    # Scale dataset

    X_scaled = scaler.transform(X)

    # Calculate inertia for K = 2 to 10

    inertia = []

    for k in range(2, 11):

        temp_kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        temp_kmeans.fit(X_scaled)

        inertia.append(
            temp_kmeans.inertia_
        )

    # Create graph

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        range(2, 11),
        inertia,
        marker="o",
        linewidth=2
    )

    # Mark K = 5

    ax.axvline(
        x=5,
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        "Elbow Method for Optimal Number of Clusters",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.set_xticks(
        range(2, 11)
    )

    ax.grid(
        True,
        alpha=0.3
    )

    st.pyplot(fig)

    st.success(
        "✅ Optimal Number of Clusters (K) = 5"
    )

    st.markdown("### 📌 Interpretation")

    st.write(
        """
        The inertia decreases as the number of clusters increases.
        The decrease becomes slower around K = 5, forming the elbow point.
        Therefore, 5 clusters are selected for the K-Means algorithm.
        """
    )


# =====================================================
# CLUSTER ANALYSIS PAGE
# =====================================================

elif page == "📊 Cluster Analysis":

    st.markdown("## 📊 Cluster Analysis")

    st.write(
        """
        The following table shows the average characteristics
        of customers in each cluster.
        """
    )

    cluster_summary = (
        df.groupby("Cluster")[
            [
                "Age",
                "Annual Income ($)",
                "Spending Score (1-100)",
                "Work Experience",
                "Family Size"
            ]
        ]
        .mean()
        .round(2)
    )

    cluster_summary["Segment"] = [
        segment_names[i]
        for i in cluster_summary.index
    ]

    st.markdown(
        "### 📈 Average Customer Characteristics"
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.markdown(
        "### 👥 Customers in Each Cluster"
    )

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Number of Customers"
    ]

    cluster_counts["Segment"] = (
        cluster_counts["Cluster"]
        .map(segment_names)
    )

    st.dataframe(
        cluster_counts,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About the Project")

    st.markdown(
        """
        ### 🎯 Customer Segmentation Using Clustering

        This project uses **K-Means Clustering** to identify groups
        of customers with similar characteristics and behaviour.

        ### 🔧 Technologies Used

        • Python  
        • Pandas  
        • NumPy  
        • Scikit-learn  
        • Matplotlib  
        • Seaborn  
        • Streamlit  
        • Joblib  

        ### 📌 Selected Features

        • Age  
        • Annual Income  
        • Spending Score  
        • Work Experience  
        • Family Size  

        ### 🤖 Machine Learning Model

        **Algorithm:** K-Means Clustering

        **Optimal K:** 5

        **Scaling:** StandardScaler

        ### 💼 Business Purpose

        Customer segmentation can help businesses understand
        customer behaviour and create targeted marketing strategies.
        """
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>Customer Segmentation Using Clustering</b><br>

    K-Means Clustering • Machine Learning Project<br>

    B.Tech CSE | Aditya University

    </div>
    """,
    unsafe_allow_html=True
)