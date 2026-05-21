import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Insurance Risk Prediction System", page_icon="🚗",layout="wide")

# =========================================================
# LOGIN SESSION
# =========================================================

if 'login_status' not in st.session_state:
    st.session_state.login_status = False

# =========================================================
# LOGIN FUNCTION
# =========================================================

def login():

    st.title("🔐 Insurance System Login")

    username = st.text_input("User ID")

    password = st.text_input("Password",type="password")

    if st.button("Login"):

        if username == "insurancesystem" and password == "admin1234":

            st.session_state.login_status = True

            st.success("Login Successful ✅")

            st.rerun()

        else:

            st.error("Invalid User ID or Password")

# =========================================================
# MAIN APP
# =========================================================

def main_app():

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv("Final_Insurance.csv")

    # CREATE RISK FLAG

    df['RISK_FLAG'] = df['RISK_ZONE'].apply(
        lambda x: 1 if x == 'High' else 0
    )

    # =====================================================
    # LOAD MODEL & SCALER
    # =====================================================

    model = joblib.load( "final_risk_model.pkl")

    scaler = joblib.load( "final_scaler.pkl")

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("🚗 Insurance Analytics")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "KPI Dashboard",
            "Analytics Dashboard",
            "Risk Prediction"
        ]
    )

    # =====================================================
    # HOME PAGE
    # =====================================================

    if page == "Home":

        st.title(
            "🚗 Insurance Risk Prediction System"
        )

        st.write(
            """
            ## 📌 Project Overview

            This application performs:

            ✔ Insurance Data Analytics  
            ✔ KPI Monitoring  
            ✔ Dashboard Visualization  
            ✔ Machine Learning Prediction  

            The Machine Learning model predicts:

            🔴 High Risk Customer  
            🟢 Low Risk Customer
            """
        )

        st.success(
            "Machine Learning Model Connected Successfully ✅"
        )

    # =====================================================
    # KPI DASHBOARD
    # =====================================================

    elif page == "KPI Dashboard":

        st.title(
            "📊 KPI Dashboard"
        )

        total_policies = len(df)

        total_claims = df['CLAIM_STATUS' ].count()

        avg_premium = round(
            df['ANNUAL_PREMIUM_NUMERIC'].mean(),2)

        avg_age = round(
            df['CLIENT_AGE' ].mean(),2)

        total_vehicles = df['BRAND'].count()

        rejected_claims = len(
            df[df['CLAIM_STATUS'] == 'Rejected'])

        claim_rejection_rate = round((rejected_claims /total_claims) * 100,2)

        accuracy_score = 83

        roc_auc_score = 0.90

        # =================================================
        # KPI ROW 1
        # =================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric("📑 Total Policies",total_policies )

        with col2:

            st.metric("📋 Total Claims",total_claims)

        with col3:

            st.metric("💰 Avg Premium",avg_premium)

        with col4:

            st.metric("👤 Avg Age",avg_age)

        # =================================================
        # KPI ROW 2
        # =================================================

        col5, col6, col7, col8 = st.columns(4)

        with col5:

            st.metric("🚗 Total Vehicles",total_vehicles)

        with col6:

            st.metric("❌ Claim Rejection %",f"{claim_rejection_rate}%")

        with col7:

            st.metric("🎯 Accuracy Score",f"{accuracy_score}%")

        with col8:

            st.metric("📈 ROC-AUC Score",roc_auc_score)

    # =====================================================
    # ANALYTICS DASHBOARD
    # =====================================================

    elif page == "Analytics Dashboard":

        st.title(
            "📊 Insurance Analytics Dashboard"
        )

        # =================================================
        # ROW 1
        # =================================================

        col1, col2 = st.columns(2)

        # =================================================
        # RISK ZONE DISTRIBUTION
        # =================================================

        with col1:

            st.subheader(
                "⚠ Risk Zone Distribution"
            )

            fig1, ax1 = plt.subplots(
                figsize=(7,5)
            )

            ax = sns.countplot(
                x='RISK_ZONE',
                data=df,
                ax=ax1
            )

            ax.bar_label(
                ax.containers[0]
            )

            plt.title(
                'Risk Zone Distribution'
            )

            st.pyplot(fig1)

        # =================================================
        # CLAIM STATUS DISTRIBUTION
        # =================================================

        with col2:

            st.subheader(
                "📋 Claim Status Distribution"
            )

            claim_df = df[
                df['CLAIM_STATUS'] != 'No Claim'
            ]

            fig2, ax2 = plt.subplots(
                figsize=(8,5)
            )

            sns.countplot(
                x='CLAIM_STATUS',
                data=claim_df,
                ax=ax2
            )

            plt.title(
                'Claim Status Distribution'
            )

            st.pyplot(fig2)

        # =================================================
        # ROW 2
        # =================================================

        col3, col4 = st.columns(2)

        # =================================================
        # PRODUCT DISTRIBUTION
        # =================================================

        with col3:

            st.subheader(
                "📦 Product Distribution"
            )

            fig3, ax3 = plt.subplots(
                figsize=(6,6)
            )

            df['PRODUCT'].value_counts().plot(
                kind='pie',
                autopct='%1.1f%%',
                ax=ax3
            )

            plt.title(
                'Product Distribution'
            )

            plt.ylabel('')

            st.pyplot(fig3)

        # =================================================
        # BRAND DAMAGE ANALYSIS
        # =================================================

        with col4:

            st.subheader(
                "🚗 Brand Damage Analysis"
            )

            top_brands = df.groupby(
                'BRAND'
            )['DAMAGE_AMOUNT_NUMERIC'].sum().sort_values(
                ascending=False
            ).head(10)

            fig4, ax4 = plt.subplots(
                figsize=(10,5)
            )

            top_brands.plot(
                kind='barh',
                ax=ax4
            )

            plt.title(
                'Brands with Highest Damage Amount'
            )

            plt.ylabel(
                'Total Damage Amount'
            )

            plt.xticks(
                rotation=45
            )

            st.pyplot(fig4)

        st.success(
            "Dashboard Loaded Successfully ✅"
        )

    # =====================================================
    # RISK PREDICTION
    # =====================================================

    elif page == "Risk Prediction":

        st.title("🤖 Insurance Risk Prediction")

        st.write("Enter customer details below.")

        # =================================================
        # USER INPUTS
        # =================================================

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        premium = st.number_input(
            "Annual Premium",
            min_value=0.0,
            value=50000.0
        )

        product = st.selectbox(
            "Product",
            [
                "Auto",
                "Health",
                "Life"
            ]
        )

        channel = st.selectbox(
            "Channel",
            [
                "Agent",
                "Online",
                "Branch"
            ]
        )

        brand = st.selectbox(
            "Brand",
            [
                "Toyota",
                "Honda",
                "BMW"
            ]
        )

        power = st.number_input(
            "Vehicle Power HP",
            min_value=0.0,
            value=120.0
        )

        current_value = st.number_input(
            "Current Vehicle Value",
            min_value=0.0,
            value=500000.0
        )

        # =================================================
        # MANUAL ENCODING
        # =================================================

        gender_map = {
            "Male": 1,
            "Female": 0
        }

        product_map = {
            "Auto": 0,
            "Health": 1,
            "Life": 2
        }

        channel_map = {
            "Agent": 0,
            "Online": 1,
            "Branch": 2
        }

        brand_map = {
            "Toyota": 0,
            "Honda": 1,
            "BMW": 2
        }

        gender = gender_map[gender]

        product = product_map[product]

        channel = channel_map[channel]

        brand = brand_map[brand]

        # =================================================
        # INPUT DATAFRAME
        # =================================================

        input_data = pd.DataFrame([[
            age,
            gender,
            premium,
            product,
            channel,
            brand,
            power,
            current_value
        ]], columns=[
            'CLIENT_AGE',
            'GENDER',
            'ANNUAL_PREMIUM_NUMERIC',
            'PRODUCT',
            'CHANNEL',
            'BRAND',
            'POWER_HP',
            'CURRENT_VALUE_NUMERIC'
        ])

        # =================================================
        # SCALE INPUT
        # =================================================

        input_scaled = scaler.transform(
            input_data
        )

        # =================================================
        # PREDICTION
        # =================================================

        if st.button(
            "Predict Risk"
        ):

            probability = model.predict_proba(
                input_scaled
            )

            risk_probability = probability[0][1]

            # CUSTOM THRESHOLD

            if risk_probability >= 0.70:

                st.error(
                    "🔴 High Risk Customer"
                )

                st.write(
                    f"Risk Probability: {risk_probability:.2f}"
                )

            else:

                st.success("🟢 Low Risk Customer")

                st.write(f"Risk Probability: {risk_probability:.2f}")

# =========================================================
# RUN APP
# =========================================================

if st.session_state.login_status:

    main_app()

else:

    login()