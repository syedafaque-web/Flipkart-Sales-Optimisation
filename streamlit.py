import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Flipkart Sales Dashboard", layout="wide", page_icon="📈")

# ---- Styling ----
st.markdown("""
<style>
[data-testid="stSidebar"] {background-image: linear-gradient(180deg, #fff7f0, #ffe9d6);} 
section.main {background: linear-gradient(180deg, #ffffff, #fffdf9);} 
.stButton>button {border-radius:12px; padding:8px 12px}
.kpi {background: linear-gradient(180deg,#ffffff,#fff9f0); border-radius:12px; padding:12px}
</style>
""", unsafe_allow_html=True)

# ---- Helper functions ----
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    # Accepts uploaded file buffer or string path
    try:
        df = pd.read_csv(path_or_buffer, low_memory=False)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return pd.DataFrame()
    return df

@st.cache_data
def preprocess(sales_df, products_df=None):
    df = sales_df.copy()
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic typed columns if exists
    for col in ["revenue", "price", "selling_price", "landing_price", "procured_quantity", "quantity", "order_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date
    date_cols = [c for c in df.columns if 'date' in c]
    if date_cols:
        try:
            df['order_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        except:
            df['order_date'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
    else:
        df['order_date'] = pd.NaT

    # Derive revenue if missing
    if 'revenue' not in df.columns and 'selling_price' in df.columns and 'procured_quantity' in df.columns:
        df['revenue'] = df['selling_price'] * df['procured_quantity']

    # Basic categories
    if 'l1_category' not in df.columns:
        for c in ['category', 'l1', 'l1_category']:
            if c in df.columns:
                df['l1_category'] = df[c]
                break
    if 'city_name' not in df.columns:
        for c in ['city', 'city_name', 'shipping_city']:
            if c in df.columns:
                df['city_name'] = df[c]
                break

    return df

# ---- Sidebar: Data sources & controls ----
st.sidebar.header("Data & settings")
use_sample = st.sidebar.checkbox("Use sample demo data (small)", value=False)
uploaded_sales = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"]) 
uploaded_products = st.sidebar.file_uploader("Upload Products CSV (optional)", type=["csv"]) 

# Provide quick local paths (user path available)
st.sidebar.markdown("""
/Users/syedafaque/Downloads/Sales.csv  
/Users/syedafaque/Downloads/products.csv  
(Or use the uploaded file above)
""")
st.sidebar.markdown("/Users/syedafaque/Downloads/Sales.csv  
/Users/syedafaque/Downloads/products.csv  
(Or use the uploaded file above)")

# ---- Load data ----
if use_sample:
    # create a small synthetic sample
    n = 1000
    rng = np.random.default_rng(42)
    sample = pd.DataFrame({
        'order_id': np.arange(n),
        'order_date': pd.date_range('2025-01-01', periods=n, freq='H'),
        'city_name': rng.choice(['Mumbai','Bengaluru','Delhi','Kolkata','Hyderabad'], n),
        'l1_category': rng.choice(['Electronics','Clothing','Home','Beauty'], n),
        'procured_quantity': rng.integers(1,5,n),
        'selling_price': rng.integers(200,5000,n)
    })
    sample['revenue'] = sample['procured_quantity'] * sample['selling_price']
    sales_df = sample
    products_df = pd.DataFrame()
else:
    if uploaded_sales is not None:
        sales_df = load_csv(uploaded_sales)
    else:
        # try typical local path
        local_path = Path('/Users/syedafaque/Downloads/Sales.csv')
        if local_path.exists():
            sales_df = load_csv(str(local_path))
        else:
            sales_df = pd.DataFrame()

    if uploaded_products is not None:
        products_df = load_csv(uploaded_products)
    else:
        local_products = Path('/Users/syedafaque/Downloads/products.csv')
        if local_products.exists():
            products_df = load_csv(str(local_products))
        else:
            products_df = pd.DataFrame()

if sales_df.empty:
    st.warning("No sales data loaded yet — upload a Sales CSV or enable sample data in the sidebar.")
    st.stop()

# Preprocess
df = preprocess(sales_df, products_df)

# ---- Main layout ----
st.title("📈 Flipkart Sales — Interactive Dashboard")
st.markdown("Quick, beautiful interface to explore revenue, products, city and category performance.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
    st.metric("Total Revenue", f"₹ {total_revenue:,.0f}")
with col2:
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    st.metric("Total Orders", f"{total_orders:,}")
with col3:
    aov = (total_revenue / total_orders) if total_orders else 0
    st.metric("AOV (Avg Order Value)", f"₹ {aov:,.1f}")
with col4:
    total_qty = df['procured_quantity'].sum() if 'procured_quantity' in df.columns else df.get('quantity', pd.Series([0])).sum()
    st.metric("Total Units Sold", f"{int(total_qty):,}")

# Filters row
st.markdown("---")
left, right = st.columns([1,3])
with left:
    cats = df['l1_category'].fillna('Unknown').unique().tolist() if 'l1_category' in df.columns else []
    sel_cats = st.multiselect("Category", options=sorted(cats), default=sorted(cats)[:3])
    cities = df['city_name'].fillna('Unknown').unique().tolist() if 'city_name' in df.columns else []
    sel_cities = st.multiselect("City", options=sorted(cities), default=sorted(cities)[:3])
    date_min = pd.to_datetime(df['order_date'].min()) if not df['order_date'].isna().all() else None
    date_max = pd.to_datetime(df['order_date'].max()) if not df['order_date'].isna().all() else None
    date_range = st.date_input("Order date range", value=(date_min.date() if date_min is not None else None, date_max.date() if date_max is not None else None))

with right:
    st.header("Trends & breakdowns")
    # Apply filters
    mask = pd.Series(True, index=df.index)
    if sel_cats:
        mask &= df['l1_category'].fillna('Unknown').isin(sel_cats)
    if sel_cities:
        mask &= df['city_name'].fillna('Unknown').isin(sel_cities)
    if date_min is not None and date_range[0] is not None:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        mask &= (df['order_date'] >= start) & (df['order_date'] <= end)
    filtered = df[mask]

    # Time series
    if 'order_date' in filtered.columns and not filtered['order_date'].isna().all():
        timeseries = (
            filtered.set_index('order_date')
            .resample('D')['revenue']
            .sum()
            .reset_index()
        )
        fig_ts = px.line(timeseries, x='order_date', y='revenue', title='Daily revenue')
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No order_date column with usable dates to draw a timeseries.")

# Two column breakdowns
c1, c2 = st.columns(2)
with c1:
    st.subheader("Top categories by revenue")
    if 'l1_category' in filtered.columns and 'revenue' in filtered.columns:
        cat_rev = filtered.groupby('l1_category')['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(10)
        fig_cat = px.bar(cat_rev, x='l1_category', y='revenue', title='Top categories')
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.write("Category or revenue columns missing.")
with c2:
    st.subheader("Top cities by revenue")
    if 'city_name' in filtered.columns and 'revenue' in filtered.columns:
        city_rev = filtered.groupby('city_name')['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(10)
        fig_city = px.bar(city_rev, x='city_name', y='revenue', title='Top cities')
        st.plotly_chart(fig_city, use_container_width=True)

# Table and download
st.markdown("---")
st.subheader("Data preview")
st.dataframe(filtered.head(500))

@st.cache_data
def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

csv_bytes = to_csv_bytes(filtered)
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_sales.csv", mime='text/csv')

# Footer with run instructions
st.markdown("---")
st.caption("Tip: Run locally with `streamlit run streamlit_app.py`. The app will try to auto-load /Users/syedafaque/Downloads/Sales.csv if present.")
