import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Flipkart Sales & Profit Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

OUT_DIR = Path("./analysis_outputs")

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_name):
    """Loads a CSV file from the output directory and caches it."""
    path = OUT_DIR / file_name
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Standardize column names for ease of use
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def format_crore(value):
    """Formats a number in Crore for display."""
    if pd.isna(value):
        return "-"
    return f"₹ {value:,.2f} Cr"

def format_aov(value):
    """Formats Average Order Value."""
    if pd.isna(value):
        return "-"
    return f"₹ {value:,.0f}"

# --- SIDEBAR & INTRO ---
with st.sidebar:
    st.title("🛒 Flipkart Analytics Engine")
    st.caption("Profit Maximization & Scalability via Chunked Data Processing")
    st.markdown("---")
    
    # Check if essential files exist
    required_files = [
        "city_summary_chunked_approx.csv",
        "category_profitability.csv",
        "monthly_revenue_comparison_table.csv",
    ]
    
    status_icon = "🟢"
    for f in required_files:
        if not (OUT_DIR / f).exists():
            status_icon = "🔴"
            st.error(f"Missing file: {f}")
    
    st.info(f"{status_icon} Data Status: {'All files loaded' if status_icon == '🟢' else 'Check analysis_outputs'}")
    st.markdown("---")

    page = st.selectbox("Navigate Dashboard", [
        "1. Executive Summary & KPIs",
        "2. Profitability Deep Dive",
        "3. Geographical & Clustering Analysis",
        "4. Time Series Forecasting",
        "5. Pricing & Anomalies"
    ])
    st.markdown("---")

# --- DATA LOADING ---
city_df = load_data("city_summary_chunked_approx.csv")
cat_df = load_data("category_profitability.csv")
prod_df = load_data("product_profitability.csv")
brand_df = load_data("brand_profitability.csv")
fc_df = load_data("monthly_revenue_comparison_table.csv")
anom_df = load_data("pricing_anomalies_sample.csv")
disc_df = load_data("discount_bins_summary.csv")

# --- DATA PREP (FOR CONSISTENCY) ---
if not city_df.empty:
    city_df['revenue_crore'] = city_df['revenue'] / 1e7
    city_df['profit_crore'] = city_df['profit'] / 1e7
    city_df.rename(columns={'avg_order_value_approx': 'aov'}, inplace=True)

# --- MAIN DASHBOARD PAGES ---

st.title(f"📈 {page}")

# ==========================================================
# 1. Executive Summary & KPIs
# ==========================================================
if page == "1. Executive Summary & KPIs":
    
    st.header("Overall Performance Snapshot")
    
    total_revenue_cr = city_df['revenue_crore'].sum() if not city_df.empty else 0
    total_profit_cr = city_df['profit_crore'].sum() if not city_df.empty else 0
    total_qty = city_df['total_qty'].sum() if not city_df.empty else 0
    avg_aov = city_df['aov'].mean() if not city_df.empty and 'aov' in city_df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Total Revenue")
        st.markdown(f"## {format_crore(total_revenue_cr)}")
        st.caption(f"Based on {total_qty/1e7:,.2f} Cr Units Sold")
    with col2:
        st.subheader("Total Profit")
        st.markdown(f"## {format_crore(total_profit_cr)}")
        
        # Calculate Margin
        margin = (total_profit_cr / total_revenue_cr) * 100 if total_revenue_cr > 0 else 0
        st.caption(f"Profit Margin: **{margin:,.2f}%**")
    with col3:
        st.subheader("Avg. Order Value (AOV)")
        st.markdown(f"## {format_aov(avg_aov)}")
        st.caption("Metric for Customer Segment Health")
    with col4:
        st.subheader("Top Category (Profit)")
        if not cat_df.empty and 'profit_crore' in cat_df.columns and 'l1_category' in cat_df.columns:
            top_cat_row = cat_df.sort_values('profit_crore', ascending=False).iloc[0]
            st.markdown(f"## {top_cat_row['l1_category']}")
            st.caption(f"Profit: {format_crore(top_cat_row['profit_crore'])}")
        else:
            st.markdown("## N/A")

    st.markdown("---")
    
    # Chart: Top 10 Cities by Revenue (Interactive Plotly)
    if not city_df.empty:
        st.subheader("Top 10 Revenue Contributors")
        top10_rev = city_df.sort_values('revenue_crore', ascending=False).head(10)
        fig_city = px.bar(
            top10_rev,
            x='city_name',
            y='revenue_crore',
            title='Top 10 Cities by Revenue (₹ Crore)',
            color='revenue_crore',
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={'revenue_crore': 'Revenue (₹ Crore)', 'city_name': 'City'},
            template="plotly_white"
        )
        st.plotly_chart(fig_city, use_container_width=True)

# ==========================================================
# 2. Profitability Deep Dive
# ==========================================================
elif page == "2. Profitability Deep Dive":

    st.header("Product, Category, and Brand Health")
    
    tab1, tab2, tab3 = st.tabs(["Category Profit", "Brand Profit", "Product Review"])
    
    with tab1:
        if not cat_df.empty and 'profit_crore' in cat_df.columns:
            st.subheader("Top/Bottom 10 Categories by Profit")
            
            top10_cat = cat_df.head(10)
            
            # Category Profit Bar Chart
            fig_cat = px.bar(
                top10_cat,
                x='l1_category',
                y='profit_crore',
                title='Top 10 Categories by Profit (₹ Crore)',
                color='profit_crore',
                color_continuous_scale=px.colors.sequential.Teal,
                template="plotly_white"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            with st.expander("Category Profitability Data"):
                st.dataframe(cat_df[['l1_category','revenue_crore','profit_crore','profit_margin']].style.format({
                    'revenue_crore': format_crore, 
                    'profit_crore': format_crore, 
                    'profit_margin': "{:.2%}"
                }))
        else:
            st.warning("Category profitability data not found.")
            
    with tab2:
        if not brand_df.empty and 'profit_crore' in brand_df.columns:
            st.subheader("Top 10 Brands by Profit")
            
            top10_brand = brand_df.head(10)
            
            # Brand Profit Bar Chart
            fig_brand = px.bar(
                top10_brand,
                x='brand_name',
                y='profit_crore',
                title='Top 10 Brands by Profit (₹ Crore)',
                color='profit_crore',
                color_continuous_scale=px.colors.sequential.Sunset,
                template="plotly_white"
            )
            st.plotly_chart(fig_brand, use_container_width=True)
            
            with st.expander("Brand Profitability Data"):
                st.dataframe(brand_df[['brand_name','revenue_crore','profit_crore','profit_margin']].style.format({
                    'revenue_crore': format_crore, 
                    'profit_crore': format_crore, 
                    'profit_margin': "{:.2%}"
                }))
        else:
            st.warning("Brand profitability data not found.")

    with tab3:
        if not prod_df.empty and 'profit_crore' in prod_df.columns:
            st.subheader("Product-Level Review")
            
            col_t3_1, col_t3_2 = st.columns(2)
            
            # Top Products
            with col_t3_1:
                st.caption("Top 15 Products by Profit")
                top15_prod = prod_df[['product_name', 'profit_crore', 'profit_margin']].head(15)
                st.dataframe(top15_prod.style.format({
                    'profit_crore': format_crore, 
                    'profit_margin': "{:.2%}"
                }), use_container_width=True)
            
            # Loss-Making Products
            with col_t3_2:
                st.caption("Top 15 Loss-Making Products (Profit < ₹0)")
                loss_prod = prod_df[prod_df['profit'] < 0].sort_values('profit').head(15)
                loss_prod['profit_crore_abs'] = abs(loss_prod['profit_crore'])
                loss_prod_display = loss_prod[['product_name', 'profit_crore_abs', 'profit_margin']].rename(columns={'profit_crore_abs': 'Loss (Cr)'})
                st.dataframe(loss_prod_display.style.format({
                    'Loss (Cr)': format_crore, 
                    'profit_margin': "{:.2%}"
                }), use_container_width=True)

            # Profit Margin Distribution Plot
            fig_margin = px.histogram(
                prod_df.dropna(subset=['profit_margin']),
                x='profit_margin',
                nbins=50,
                title='Distribution of Product Profit Margins',
                height=400,
                template="plotly_white"
            )
            fig_margin.update_layout(xaxis_title="Profit Margin (Ratio)")
            st.plotly_chart(fig_margin, use_container_width=True)
        else:
            st.warning("Product profitability data not found.")


# ==========================================================
# 3. Geographical & Clustering Analysis
# ==========================================================
elif page == "3. Geographical & Clustering Analysis":
    
    st.header("City Performance & Segmentation")
    
    if not city_df.empty and 'cluster' in city_df.columns:
        
        st.markdown(
            """
            The **K-Means Clustering (n=4)** segmentation is based on normalized **Revenue**, **Profit**, and **Avg. Order Value (AOV)** to identify distinct city profiles for targeted marketing and logistics.
            """
        )
        
        # City Clustering Scatter Plot
        fig_cluster = px.scatter(
            city_df.head(500), # Sample for visualization speed
            x='revenue_crore',
            y='aov',
            color='cluster',
            size='profit_crore',
            hover_data=['city_name', 'profit_crore'],
            title='City Segmentation: Revenue vs. AOV (Size = Profit)',
            template="plotly_white"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("Cluster Profiles (Mean Values)")
        
        # Cluster Profile Table
        cluster_profile = city_df.groupby('cluster')[['revenue_crore', 'aov', 'profit_crore']].mean().sort_values('revenue_crore', ascending=False)
        cluster_profile.columns = ['Avg. Revenue (Cr)', 'Avg. AOV (₹)', 'Avg. Profit (Cr)']
        
        st.dataframe(cluster_profile.style.format({
            'Avg. Revenue (Cr)': format_crore, 
            'Avg. AOV (₹)': format_aov, 
            'Avg. Profit (Cr)': format_crore
        }))
        
        st.markdown(
            """
            *Interpretation:* **Cluster 1 (High Revenue/Profit)** likely represents major metro hubs. 
            **Cluster 0 (Moderate/High AOV)** could indicate smaller cities with high-value buyers. 
            Strategies should be tailored: **Retention for Cluster 1, AOV-boosting for Cluster 0.**
            """
        )
    else:
        st.warning("City summary or clustering data not found. Please ensure the analysis script ran correctly.")

# ==========================================================
# 4. Time Series Forecasting
# ==========================================================
elif page == "4. Time Series Forecasting":
    
    st.header("Monthly Revenue Trend & ARIMA Forecast")
    
    monthly_hist_df = load_data("monthly_revenue_series_crore.csv") # Used for full history
    
    if not fc_df.empty and not monthly_hist_df.empty:
        
        # Standardize column names
        fc_df.rename(columns={
            'actual_revenue_crore': 'Actual',
            'predicted_revenue_crore': 'Predicted'
        }, inplace=True)
        
        # --- Prepare data for plotting ---
        # History (from monthly_hist_df)
        hist_df = monthly_hist_df.copy()
        hist_df.columns = ['Month', 'Revenue']
        hist_df['Type'] = 'Actual'
        hist_df['Month'] = pd.to_datetime(hist_df['Month'], errors='coerce')

        # Forecast (from fc_df)
        forecast_subset = fc_df[['month', 'predicted']].dropna().copy()
        forecast_subset.rename(columns={'predicted': 'Revenue', 'month': 'Month'}, inplace=True)
        forecast_subset['Type'] = 'Forecast'
        forecast_subset['Month'] = pd.to_datetime(forecast_subset['Month'], errors='coerce')

        # Combine data for plotting
        plot_data = pd.concat([hist_df, forecast_subset]).sort_values('Month').drop_duplicates(subset=['Month'], keep='last')
        
        # Plotly Line Chart for History and Forecast
        fig_ts = px.line(
            plot_data,
            x='Month',
            y='Revenue',
            color='Type',
            line_dash='Type',
            title='Monthly Revenue History and ARIMA (1,1,1) Forecast (₹ Crore)',
            markers=True,
            template="plotly_white"
        )
        fig_ts.update_traces(line=dict(dash='solid', width=3), selector=dict(name='Actual'))
        fig_ts.update_traces(line=dict(dash='dot', width=3), marker=dict(color='red'), selector=dict(name='Forecast'))
        fig_ts.update_yaxes(title_text="Revenue (₹ Crore)")
        
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")

        # Comparison Table
        st.subheader("Revenue Comparison: Last 6 Actual vs. Next 6 Forecast")
        
        comparison_cols = ['month', 'Actual', 'Predicted']
        fc_display = fc_df[comparison_cols].copy()
        
        # Format for display
        fc_display['Actual'] = fc_display['Actual'].apply(lambda x: format_crore(x) if pd.notna(x) else '-')
        fc_display['Predicted'] = fc_display['Predicted'].apply(lambda x: format_crore(x) if pd.notna(x) else '-')
        
        st.table(fc_display.style)
        
    else:
        st.warning("Monthly revenue series or forecast comparison data not found.")

# ==========================================================
# 5. Pricing & Anomalies
# ==========================================================
elif page == "5. Pricing & Anomalies":
    
    st.header("Discount Optimization and Health Check")

    tab1, tab2 = st.tabs(["Discount Analysis", "Pricing Anomalies"])
    
    with tab1:
        if not disc_df.empty:
            st.subheader("Revenue Contribution by Discount Bin")
            
            # Use 'revenue_crore' from your final analysis file
            disc_df['revenue_crore'] = disc_df['revenue'] / 1e7
            
            # Ensure correct order for bins
            order = ['0%','0-5%','5-10%','10-20%','20-50%','50-100%','>100%']
            disc_df['bin_order'] = disc_df['disc_bin'].astype(str).map({v:i for i,v in enumerate(order)})
            disc_summary_sorted = disc_df.sort_values('bin_order').dropna(subset=['disc_bin'])

            # Plot Revenue vs Discount Bin
            fig_disc_rev = px.bar(
                disc_summary_sorted,
                x='disc_bin',
                y='revenue_crore',
                title='Revenue by Discount Bin (₹ Crore)',
                color='revenue_crore',
                category_orders={"disc_bin": order},
                color_continuous_scale=px.colors.sequential.Viridis,
                template="plotly_white"
            )
            st.plotly_chart(fig_disc_rev, use_container_width=True)
            
            st.info(
                """
                **Optimization Insight:** The highest revenue often comes from the lowest discount bins. 
                Focus on the **5%-20%** bins to determine the optimal discount that maximizes volume without eroding profit too heavily.
                """
            )

        else:
            st.warning("Discount bin summary data not found.")

    with tab2:
        if not anom_df.empty:
            st.subheader("Pricing Errors and High Discount Risks")
            
            st.markdown(
                """
                Rows flagged where **Unit Selling Price < Landing Cost** OR **Discount Rate > 70%**. 
                These are critical operational losses that need immediate review.
                """
            )
            
            # Anomalies Scatter Plot
            if all(c in anom_df.columns for c in ['landing_cost_per_unit', 'unit_selling_price']):
                fig_anom = px.scatter(
                    anom_df.head(500), 
                    x='landing_cost_per_unit',
                    y='unit_selling_price',
                    color='discount_rate',
                    title='Selling Price vs. Landing Cost (Identified Anomalies)',
                    hover_data=['product_name', 'discount_rate'],
                    template="plotly_white"
                )
                # Add 45-degree line (y=x) where SP = LC (i.e., zero gross profit)
                min_val = min(anom_df['landing_cost_per_unit'].min(), anom_df['unit_selling_price'].min())
                max_val = max(anom_df['landing_cost_per_unit'].max(), anom_df['unit_selling_price'].max())
                
                fig_anom.add_shape(
                    type='line',
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color='Red', width=2, dash='dash')
                )
                st.plotly_chart(fig_anom, use_container_width=True)

            # Anomalies Table
            st.caption(f"Sampled {len(anom_df)} Critical Anomalies")
            anom_display = anom_df[['product_name', 'unit_selling_price', 'landing_cost_per_unit', 'discount_rate']].head(50)
            st.dataframe(anom_display.style.format({
                'unit_selling_price': "₹ {:,.2f}", 
                'landing_cost_per_unit': "₹ {:,.2f}", 
                'discount_rate': "{:.1%}"
            }))
        else:
            st.warning("Pricing anomalies sample data not found.")
