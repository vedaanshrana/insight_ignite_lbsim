import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adobe Customer Insights Commander",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Crazy" Visual Appeal
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1rem;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #FAFAFA; 
    }
    .stPlotlyChart {
        background-color: #0E1117;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. ROBUST DATA GENERATION / LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Tries to load 'Adobe_Dataset.csv'. If not found, generates a massive
    synthetic dataset covering ALL objectives (Conversion, Usage, Churn, Plans, CLV).
    """
    filename = 'Adobe_Dataset.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print("CSV file not found")

df = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Team A")
#st.logo("dare_lbsim.jpg", size='large')
st.sidebar.write("Presented By: Shiwang Gupta and Vedaansh Rana")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to:", [
    "1. Executive Overview",
    "2. Conversion Prediction",
    "3. Churn Radar",
    "4. Marketing Optimization",
    "5. Plan Recommendations",
    "6. CLV & Satisfaction"
])
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the sidebar controls in each section to simulate scenarios.")

# -----------------------------------------------------------------------------
# 4. DASHBOARD PAGES
# -----------------------------------------------------------------------------

# === PAGE 1: EXECUTIVE OVERVIEW ===
if page == "1. Executive Overview":
    st.title("📊 Executive Command Center")
    st.markdown("### High-Level Performance Metrics")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_rev = (df['Lifetime_Value'].sum() / 1e6)
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">${total_rev:.2f}M</div><div class="metric-label">Total Lifetime Value</div></div>',
            unsafe_allow_html=True)
    with col2:
        conv_rate = (df['Conversion_Status'].mean() * 100)
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{conv_rate:.1f}%</div><div class="metric-label">Conversion Rate</div></div>',
            unsafe_allow_html=True)
    with col3:
        churn_rate = (df['Churn_Status'].mean() * 100)
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color: #FF5252;">{churn_rate:.1f}%</div><div class="metric-label">Churn Rate</div></div>',
            unsafe_allow_html=True)
    with col4:
        avg_sat = df['Satisfaction_Score'].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{avg_sat:.1f}/10</div><div class="metric-label">Avg Satisfaction</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # Overview Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌍 Revenue by Region")
        fig_map = px.sunburst(df, path=['Region', 'Industry'], values='Lifetime_Value',
                              color='Lifetime_Value', color_continuous_scale='viridis')
        st.plotly_chart(fig_map, use_container_width=True)

    with c2:
        st.subheader("📉 Churn by Industry")
        churn_by_ind = df.groupby('Industry')['Churn_Status'].mean().reset_index()
        fig_churn = px.bar(churn_by_ind, x='Industry', y='Churn_Status', color='Churn_Status',
                           color_continuous_scale='Reds', title="Churn Rate by Industry")
        st.plotly_chart(fig_churn, use_container_width=True)

# === PAGE 2: CONVERSION PREDICTION ===
elif page == "2. Conversion Prediction":
    st.title("🔮 Conversion Prediction Engine")

    # Train Model on the fly
    features = ['Engagement_Score', 'Features_Used_Count', 'Days_Active_in_Trial', 'Login_Frequency_per_Week']
    X = df[features]
    y = df['Conversion_Status']
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🧪 Simulation Lab")
        st.markdown("Adjust user behavior to predict conversion probability:")
        eng = st.slider("Engagement Score", 0, 100, 50)
        feat = st.slider("Features Used", 0, 25, 5)
        days = st.slider("Days Active", 0, 30, 7)
        login = st.slider("Logins per Week", 0, 50, 3)

        # Predict
        input_data = pd.DataFrame([[eng, feat, days, login]], columns=features)
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("### Probability of Conversion:")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Conversion %"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "lime" if prob > 0.5 else "red"}}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=0, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("🧠 What Drives Conversion?")
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig_imp = px.bar(imp, x=imp.values, y=imp.index, orientation='h',
                         title="Feature Importance", color=imp.values, color_continuous_scale='Teal')
        st.plotly_chart(fig_imp, use_container_width=True)

# === PAGE 3: CHURN RADAR ===
elif page == "3. Churn Radar":
    st.title("🚨 Churn Risk Radar")

    # 3D Visualization of Risk
    st.subheader("Dimensions of Churn")
    st.markdown("Explore how Recency (Last Login), Satisfaction, and Support Tickets interact to drive churn.")

    fig_3d = px.scatter_3d(df, x='Last_Login_Days_Ago', y='Satisfaction_Score', z='Support_Tickets_Raised',
                           color='Churn_Status', opacity=0.7,
                           color_continuous_scale=['green', 'red'],
                           title="3D Risk Map: Recency vs Satisfaction vs Support")
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

    st.info("Insight: The 'Red Zone' is clearly visible where Recency is high and Satisfaction is low.")

# === PAGE 4: MARKETING OPTIMIZATION ===
elif page == "4. Marketing Optimization":
    st.title("🎯 Marketing Channel Matrix")

    # Aggregation
    metrics = df.groupby('Marketing_Channel').agg(
        Total_Users=('Customer_ID', 'count'),
        Conversions=('Conversion_Status', 'sum'),
        Avg_CLV=('Lifetime_Value', 'mean')
    ).reset_index()
    metrics['Conv_Rate'] = (metrics['Conversions'] / metrics['Total_Users']) * 100

    # Logic for Quadrants
    avg_cr = metrics['Conv_Rate'].mean()
    avg_clv = metrics['Avg_CLV'].mean()


    def get_quadrant(row):
        if row['Conv_Rate'] >= avg_cr and row['Avg_CLV'] >= avg_clv: return "🌟 Star (Scale)"
        if row['Conv_Rate'] < avg_cr and row['Avg_CLV'] >= avg_clv: return "💎 Hidden Gem (Optimize)"
        if row['Conv_Rate'] >= avg_cr and row['Avg_CLV'] < avg_clv: return "🐄 Cash Cow (Upsell)"
        return "❓ Question Mark"


    metrics['Strategy'] = metrics.apply(get_quadrant, axis=1)

    # Interactive Scatter
    fig_matrix = px.scatter(metrics, x='Conv_Rate', y='Avg_CLV',
                            color='Strategy', size='Total_Users',
                            hover_name='Marketing_Channel', text='Marketing_Channel',
                            title="Optimization Matrix: CLV vs Conversion Rate",
                            color_discrete_map={
                                "🌟 Star (Scale)": "#00CC96",
                                "💎 Hidden Gem (Optimize)": "#AB63FA",
                                "🐄 Cash Cow (Upsell)": "#FFA15A",
                                "❓ Question Mark": "#EF553B"
                            })

    # Add quadrant lines
    fig_matrix.add_hline(y=avg_clv, line_dash="dash", annotation_text="Avg CLV")
    fig_matrix.add_vline(x=avg_cr, line_dash="dash", annotation_text="Avg Conv Rate")
    fig_matrix.update_traces(textposition='top center')
    fig_matrix.update_layout(height=600)

    st.plotly_chart(fig_matrix, use_container_width=True)

# === PAGE 5: PLAN RECOMMENDATIONS ===
elif page == "5. Plan Recommendations":
    st.title("📋 Smart Plan Recommender")

    # Train Recommender
    success_df = df[(df['Satisfaction_Score'] >= 7) & (df['Churn_Status'] == 0)]
    features = ['Features_Used_Count', 'Cloud_Storage_Usage_GB', 'Avg_Session_Time_mins']
    clf = RandomForestClassifier(n_estimators=50).fit(success_df[features], success_df['Subscription_Plan'])

    df['Recommended_Plan'] = clf.predict(df[features])

    # 1. Sankey Diagram (The "Crazy" Viz)
    st.subheader("🔄 Customer Flow: Current vs. Ideal Plan")

    # Data Prep for Sankey
    sankey_data = df.groupby(['Subscription_Plan', 'Recommended_Plan']).size().reset_index(name='count')

    # Node mapping
    all_nodes = list(pd.concat([sankey_data['Subscription_Plan'], sankey_data['Recommended_Plan']]).unique())
    node_map = {node: i for i, node in enumerate(all_nodes)}

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes, color="blue"
        ),
        link=dict(
            source=sankey_data['Subscription_Plan'].map(node_map),
            target=sankey_data['Recommended_Plan'].map(node_map),
            value=sankey_data['count']
        ))])

    fig_sankey.update_layout(title_text="Current Plan (Left) → AI Recommended Plan (Right)", font_size=12, height=500)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # 2. Upsell Table
    st.subheader("💰 Upsell Opportunities")
    upsell_mask = (df['Subscription_Plan'] == 'Basic') & (df['Recommended_Plan'].isin(['Business', 'Enterprise']))
    upsells = df[upsell_mask][
        ['Customer_ID', 'Usage_Gap' if 'Usage_Gap' in df else 'Features_Used_Count', 'Recommended_Plan']]
    st.dataframe(upsells.head(10), use_container_width=True)
    st.caption(f"Showing top 10 of {len(upsells)} customers identified for upsell.")

# === PAGE 6: CLV & SATISFACTION ===
elif page == "6. CLV & Satisfaction":
    st.title("❤️ Satisfaction & Value Drivers")

    # Regression Analysis
    reg_features = ['Support_Tickets_Raised', 'Avg_Session_Time_mins', 'Engagement_Score']
    model_sat = RandomForestRegressor(n_estimators=50).fit(df[reg_features], df['Satisfaction_Score'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 The Cost of Bad Support")
        # Simulator
        tickets = np.array([0, 1, 2, 3, 5, 7, 10])
        # Fix other vars at mean for simulation
        sim_data = pd.DataFrame({
            'Support_Tickets_Raised': tickets,
            'Avg_Session_Time_mins': [df['Avg_Session_Time_mins'].mean()] * len(tickets),
            'Engagement_Score': [df['Engagement_Score'].mean()] * len(tickets)
        })
        pred_sat = model_sat.predict(sim_data)

        fig_sim = px.line(x=tickets, y=pred_sat, markers=True,
                          labels={'x': 'Support Tickets', 'y': 'Predicted Satisfaction'},
                          title="Predicted Satisfaction vs. Ticket Volume")
        fig_sim.update_traces(line_color='#FF4B4B', line_width=4)
        st.plotly_chart(fig_sim, use_container_width=True)

    with col2:
        st.subheader("💸 Engagement = Revenue")
        fig_scatter = px.scatter(df, x='Engagement_Score', y='Lifetime_Value', color='Satisfaction_Score',
                                 color_continuous_scale='viridis', title="Engagement vs CLV")
        st.plotly_chart(fig_scatter, use_container_width=True)

# Footer
st.markdown("---")

st.markdown("🚀 **Adobe Analytics Dashboard** | Built with Python & Streamlit")
