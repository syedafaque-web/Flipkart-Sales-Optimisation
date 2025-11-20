#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0) Install & imports
# Run this cell if packages missing (remove '#' to run pip installs)
# !pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl tqdm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# In[2]:


# 1) Load files (your exact paths)
sales_path = "/Users/syedafaque/Downloads/Sales.csv"
products_path = "/Users/syedafaque/Downloads/products.csv"

# Load products fully (safe; ~32k rows)
products = pd.read_csv(products_path, low_memory=False)
products.columns = [str(c).strip() for c in products.columns]

# For Sales: load a small sample for schema/preview (do NOT load 46M rows here)
SAMPLE_ROWS = 2000
sales = pd.read_csv(sales_path, nrows=SAMPLE_ROWS, low_memory=False)
sales.columns = [str(c).strip() for c in sales.columns]

print("products.csv shape:", products.shape)
print("products.csv columns:", products.columns.tolist())
print()
print("Sales sample shape:", sales.shape)
print("Sales sample columns:", sales.columns.tolist())

# quick peek
display(products.head(2))
display(sales.head(2))

# NOTE: full Sales processing is done later in chunked cells to avoid crashing.


# In[3]:


# 2) Clean trivial Unnamed index columns
# Keep copies
sales_orig = sales.copy()
products_orig = products.copy()

for df, name in [(sales, 'sales'), (products, 'products')]:
    unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
    if unnamed:
        print(f"{name} dropping columns: {unnamed}")
        df.drop(columns=unnamed, inplace=True, errors='ignore')

# strip spaces (already done above, but keep for safety)
sales.columns = [c.strip() for c in sales.columns]
products.columns = [c.strip() for c in products.columns]

print("sales columns after drop:", sales.columns.tolist())
print("products columns after drop:", products.columns.tolist())


# In[4]:


# 3) Nulls & duplicate checks
print("Sales (sample): null counts (top 20)")
display(sales.isnull().sum().sort_values(ascending=False).head(20))

print("\nProducts: null counts (top 20)")
display(products.isnull().sum().sort_values(ascending=False).head(20))

# duplicates check (sample-level for sales)
if 'order_id' in sales.columns:
    print("sales sample duplicate order_id count:", sales['order_id'].duplicated().sum())
if 'product_id' in products.columns:
    print("products duplicate product_id count:", products['product_id'].duplicated().sum())


# In[5]:


# 4) Convert dtypes
if 'date_' in sales.columns:
    sales['date_'] = pd.to_datetime(sales['date_'], errors='coerce')
    print("date_ ->", sales['date_'].dt.date.min(), "to", sales['date_'].dt.date.max())
else:
    print("No 'date_' column found in sample.")

num_cols = ['procured_quantity','unit_selling_price','total_discount_amount','total_weighted_landing_price']
for c in num_cols:
    if c in sales.columns:
        sales[c] = pd.to_numeric(sales[c], errors='coerce')
        print(c, "->", sales[c].isnull().sum(), "nulls after numeric coercion (sample)")


# In[6]:


# 5) Landing price check (sample)
if 'total_weighted_landing_price' in sales.columns and 'procured_quantity' in sales.columns:
    mask = sales['procured_quantity'] > 0
    sales.loc[mask, 'landing_per_unit_infer'] = (
        sales.loc[mask, 'total_weighted_landing_price'] / sales.loc[mask, 'procured_quantity']
    )
    display(sales[['procured_quantity','total_weighted_landing_price','landing_per_unit_infer']].sample(min(8, len(sales))))
    display(sales['landing_per_unit_infer'].describe())
else:
    print("Landing or quantity column missing in sample.")


# In[7]:


# 6) Derived metrics (sample preview)
df = sales.copy()   # sample-level preview only

# ensure numeric columns exist (sample)
for c in ['procured_quantity','unit_selling_price','total_discount_amount','total_weighted_landing_price']:
    if c not in df.columns:
        df[c] = 0.0
    else:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# revenue & cost calculations (digit-by-digit)
df['sell_times_qty'] = df['unit_selling_price'] * df['procured_quantity']
df['revenue'] = df['sell_times_qty'] - df['total_discount_amount']

if 'landing_per_unit_infer' in df.columns:
    df['landing_cost_per_unit'] = df['landing_per_unit_infer']
else:
    df['landing_cost_per_unit'] = np.nan

df['cost_total'] = df['landing_cost_per_unit'] * df['procured_quantity']
df['profit'] = df['revenue'] - df['cost_total']
df['profit_margin'] = np.where(df['revenue']==0, np.nan, df['profit']/df['revenue'])

display(df[['unit_selling_price','procured_quantity','total_discount_amount',
            'sell_times_qty','revenue','landing_cost_per_unit',
            'cost_total','profit','profit_margin']].sample(min(8, len(df))))


# In[8]:


# 7) Merge (sample preview)
print("'product_id' in sales sample:", 'product_id' in df.columns,
      "  in products:", 'product_id' in products.columns)

if 'product_id' in df.columns and 'product_id' in products.columns:
    merged = df.merge(products, how='left', on='product_id', suffixes=('_s','_p'))
    print("merged sample shape:", merged.shape)
    display(merged[['product_id','product_name','brand_name',
                    'l0_category','l1_category','l2_category']].head(6))
else:
    print("product_id missing in sample or products; cannot merge sample.")

# NOTE: For the full Sales file, do NOT run an in-memory merge of the whole file.
# Use the chunked pipeline provided below (see cell 'CHUNKED PIPELINE (run next)').


# In[9]:


# ===== CHUNKED PIPELINE =====
SALES_PATH = "/Users/syedafaque/Downloads/Sales.csv"
CHUNKSIZE = 400_000   # reduce to 200k if memory pressure

def safe_to_numeric(s):
    return pd.to_numeric(s, errors='coerce').fillna(0)

def compute_row_metrics(chunk):
    for c in ['procured_quantity','unit_selling_price','total_discount_amount','total_weighted_landing_price']:
        if c in chunk.columns:
            chunk[c] = safe_to_numeric(chunk[c])
        else:
            chunk[c] = 0.0
    chunk['sell_times_qty'] = chunk['unit_selling_price'] * chunk['procured_quantity']
    chunk['revenue'] = chunk['sell_times_qty'] - chunk['total_discount_amount']
    chunk['landing_cost_per_unit'] = np.nan
    mask = chunk['procured_quantity'] > 0
    if mask.any():
        chunk.loc[mask, 'landing_cost_per_unit'] = chunk.loc[mask, 'total_weighted_landing_price'] / chunk.loc[mask, 'procured_quantity']
    chunk['cost_total'] = chunk['landing_cost_per_unit'].fillna(0) * chunk['procured_quantity']
    chunk['profit'] = chunk['revenue'] - chunk['cost_total']
    chunk['profit_margin'] = np.where(chunk['revenue']==0, np.nan, chunk['profit']/chunk['revenue'])
    return chunk

# Accumulators
city_acc = {}
cat_acc = {}
brand_acc = {}
prod_acc = {}
anomaly_samples = []
MAX_ANOM_SAMPLES = 500

usecols = ['date_','city_name','order_id','procured_quantity','unit_selling_price','total_discount_amount','product_id','total_weighted_landing_price']
reader = pd.read_csv(SALES_PATH, usecols=usecols, parse_dates=['date_'], chunksize=CHUNKSIZE, low_memory=True)

for chunk in tqdm(reader, desc="Processing chunks"):
    chunk = compute_row_metrics(chunk)
    # merge product master (products small)
    chunk = chunk.merge(products[['product_id','product_name','brand_name','l1_category','l0_category','l2_category']],
                        how='left', on='product_id')

    # city aggregation within chunk
    gcity = chunk.groupby('city_name').agg(revenue=('revenue','sum'), qty=('procured_quantity','sum'), rows=('revenue','count'), profit=('profit','sum'))
    for city, r in gcity.iterrows():
        rev, qty, rows, profit = float(r['revenue']), float(r['qty']), int(r['rows']), float(r['profit'])
        if city in city_acc:
            city_acc[city][0] += rev; city_acc[city][1] += qty; city_acc[city][2] += rows; city_acc[city][3] += profit
        else:
            city_acc[city] = [rev, qty, rows, profit]

    # category (l1)
    if 'l1_category' in chunk.columns:
        gcat = chunk.groupby('l1_category').agg(revenue=('revenue','sum'), qty=('procured_quantity','sum'), profit=('profit','sum'))
        for cat, r in gcat.iterrows():
            rev, qty, profit = float(r['revenue']), float(r['qty']), float(r['profit'])
            if cat in cat_acc:
                cat_acc[cat][0] += rev; cat_acc[cat][1] += qty; cat_acc[cat][2] += profit
            else:
                cat_acc[cat] = [rev, qty, profit]

    # brand
    if 'brand_name' in chunk.columns:
        gbr = chunk.groupby('brand_name').agg(revenue=('revenue','sum'), qty=('procured_quantity','sum'), profit=('profit','sum'))
        for br, r in gbr.iterrows():
            rev, qty, profit = float(r['revenue']), float(r['qty']), float(r['profit'])
            if br in brand_acc:
                brand_acc[br][0] += rev; brand_acc[br][1] += qty; brand_acc[br][2] += profit
            else:
                brand_acc[br] = [rev, qty, profit]

    # product
    gprod = chunk.groupby('product_id').agg(revenue=('revenue','sum'), qty=('procured_quantity','sum'), profit=('profit','sum'))
    prod_names = chunk.groupby('product_id')['product_name'].first().to_dict()
    for pid, r in gprod.iterrows():
        rev, qty, profit = float(r['revenue']), float(r['qty']), float(r['profit'])
        pname = prod_names.get(pid, None)
        if pid in prod_acc:
            prod_acc[pid][0] += rev; prod_acc[pid][1] += qty; prod_acc[pid][2] += profit
        else:
            prod_acc[pid] = [rev, qty, profit, pname]

    # anomaly sampling
    maskq = chunk['procured_quantity'] > 0
    if maskq.any():
        chunk.loc[maskq, 'disc_per_unit'] = chunk.loc[maskq, 'total_discount_amount'] / chunk.loc[maskq, 'procured_quantity']
        chunk['discount_rate'] = np.where(chunk['unit_selling_price']>0, chunk['disc_per_unit'] / chunk['unit_selling_price'], 0)
    else:
        chunk['discount_rate'] = 0

    cond = (chunk['unit_selling_price'] < chunk['landing_cost_per_unit']) | (chunk['discount_rate'] > 0.7)
    if cond.any() and len(anomaly_samples)*50 < MAX_ANOM_SAMPLES:
        sample = chunk.loc[cond, ['product_id','product_name','brand_name','unit_selling_price','landing_cost_per_unit','discount_rate','procured_quantity','order_id']].head(50)
        anomaly_samples.append(sample)

# Build summary DataFrames
city_summary = pd.DataFrame([(k,*v) for k,v in city_acc.items()], columns=['city_name','revenue','total_qty','num_rows','profit']).sort_values('revenue', ascending=False)
city_summary['avg_order_value_approx'] = city_summary['revenue'] / city_summary['num_rows']

category_summary = pd.DataFrame([(k,*v) for k,v in cat_acc.items()], columns=['l1_category','revenue','total_qty','profit']).sort_values('revenue', ascending=False)
category_summary['profit_margin'] = category_summary['profit'] / category_summary['revenue']

brand_summary = pd.DataFrame([(k,*v) for k,v in brand_acc.items()], columns=['brand_name','revenue','total_qty','profit']).sort_values('revenue', ascending=False)
brand_summary['profit_margin'] = brand_summary['profit'] / brand_summary['revenue']

product_summary = pd.DataFrame([(k,*v) for k,v in prod_acc.items()], columns=['product_id','revenue','total_qty','profit','product_name']).sort_values('revenue', ascending=False)

# Save summaries
city_summary.to_csv(OUT_DIR/"city_summary_chunked_approx.csv", index=False)
category_summary.to_csv(OUT_DIR/"category_summary_chunked.csv", index=False)
brand_summary.to_csv(OUT_DIR/"brand_summary_chunked.csv", index=False)
product_summary.head(200).to_csv(OUT_DIR/"top_products_chunked_200.csv", index=False)

if anomaly_samples:
    anom_df = pd.concat(anomaly_samples).drop_duplicates().reset_index(drop=True)
    anom_df.to_csv(OUT_DIR/"pricing_anomalies_sample.csv", index=False)
    print("Saved anomaly sample rows:", len(anom_df))
else:
    print("No anomaly samples captured.")

print("Chunked processing complete. Summaries saved to", OUT_DIR)


# In[10]:


# 8) City-wise summary (with crore conversion)
city_summary = pd.read_csv(OUT_DIR/"city_summary_chunked_approx.csv")

city_summary['revenue_crore'] = city_summary['revenue'] / 1e7
city_summary['total_qty_crore'] = city_summary['total_qty'] / 1e7
city_summary['profit_crore'] = city_summary['profit'] / 1e7

display(city_summary[['city_name','revenue_crore','total_qty_crore','profit_crore','avg_order_value_approx']].head(20))

# Plot top 10 cities in crore
top10 = city_summary.head(10)
plt.figure(figsize=(10,4))
plt.bar(top10['city_name'].astype(str), top10['revenue_crore'])
plt.xticks(rotation=45)
plt.title("Top 10 Cities by Revenue (Crore ₹)")
plt.ylabel("Revenue (₹ Crore)")
plt.tight_layout()
plt.show()


# In[11]:


# 9) Category summary in crores
category_summary = pd.read_csv(OUT_DIR/"category_summary_chunked.csv")

category_summary['revenue_crore'] = category_summary['revenue'] / 1e7
category_summary['total_qty_crore'] = category_summary['total_qty'] / 1e7
category_summary['profit_crore'] = category_summary['profit'] / 1e7
category_summary['profit_margin'] = category_summary['profit'] / category_summary['revenue']

category_summary = category_summary.sort_values('revenue', ascending=False)
display(category_summary.head(20))

# Plot revenue crore
plt.figure(figsize=(10,4))
plt.bar(category_summary['l1_category'].astype(str).head(10),
        category_summary['revenue_crore'].head(10))
plt.xticks(rotation=60)
plt.title("Top Categories by Revenue (Crore ₹)")
plt.ylabel("Revenue (₹ Crore)")
plt.tight_layout()
plt.show()


# In[12]:


# 10) Products & Brands with crore conversion
product_summary = pd.read_csv(OUT_DIR/"top_products_chunked_200.csv")
brand_summary = pd.read_csv(OUT_DIR/"brand_summary_chunked.csv")

product_summary['revenue_crore'] = product_summary['revenue'] / 1e7
product_summary['total_qty_crore'] = product_summary['total_qty'] / 1e7
product_summary['profit_crore'] = product_summary['profit'] / 1e7

brand_summary['revenue_crore'] = brand_summary['revenue'] / 1e7
brand_summary['total_qty_crore'] = brand_summary['total_qty'] / 1e7
brand_summary['profit_crore'] = brand_summary['profit'] / 1e7

display(product_summary.head(20))
display(brand_summary.head(20))

# Top 10 products in crore
plt.figure(figsize=(10,4))
plt.bar(product_summary['product_name'].astype(str).head(10),
        product_summary['revenue_crore'].head(10))
plt.xticks(rotation=60)
plt.title("Top 10 Products by Revenue (Crore ₹)")
plt.ylabel("Revenue (₹ Crore)")
plt.tight_layout()
plt.show()


# In[13]:


# Run this cell in your Jupyter notebook (it streams Sales.csv, writes discount_bins_summary.csv, then plots)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Paths & tuning
SALES_PATH = "/Users/syedafaque/Downloads/Sales.csv"
OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISC_PATH = OUT_DIR/"discount_bins_summary.csv"
CHUNKSIZE = 400_000   # reduce to 200_000 or 100_000 if you run out of memory

# Bins/labels
bins = [-0.01, 0, 0.05, 0.1, 0.2, 0.5, 1.0, 10]
labels = ['0%','0-5%','5-10%','10-20%','20-50%','50-100%','>100%']

# Accumulator
disc_acc = {lab: {'revenue': 0.0, 'qty': 0.0, 'profit_sum': 0.0, 'rows': 0} for lab in labels}

# Stream the Sales file and compute per-bin aggregates
usecols = ['procured_quantity','unit_selling_price','total_discount_amount']
reader = pd.read_csv(SALES_PATH, usecols=usecols, chunksize=CHUNKSIZE, low_memory=True)

print("Streaming Sales.csv and computing discount bins (this may take a while)...")
for chunk in tqdm(reader, desc="Discount streaming"):
    # numeric safety
    chunk['procured_quantity'] = pd.to_numeric(chunk['procured_quantity'], errors='coerce').fillna(0)
    chunk['unit_selling_price'] = pd.to_numeric(chunk['unit_selling_price'], errors='coerce').fillna(0)
    chunk['total_discount_amount'] = pd.to_numeric(chunk['total_discount_amount'], errors='coerce').fillna(0)
    
    # revenue and per-unit discount
    chunk['revenue'] = chunk['unit_selling_price'] * chunk['procured_quantity'] - chunk['total_discount_amount']
    mask_q = chunk['procured_quantity'] > 0
    chunk.loc[mask_q, 'disc_per_unit'] = chunk.loc[mask_q, 'total_discount_amount'] / chunk.loc[mask_q, 'procured_quantity']
    chunk['discount_rate'] = np.where(chunk['unit_selling_price']>0, chunk['disc_per_unit'].fillna(0) / chunk['unit_selling_price'], 0)
    
    # profit_sum not directly available in Sales unless you compute cost; keep profit_sum=NaN for now
    # Bin the discount_rate
    chunk['disc_bin'] = pd.cut(chunk['discount_rate'].fillna(0), bins=bins, labels=labels)
    
    # Aggregate per chunk
    g = chunk.groupby('disc_bin').agg(
        revenue_chunk = ('revenue','sum'),
        qty_chunk = ('procured_quantity','sum'),
        rows_chunk = ('revenue','count')
    )
    for lab, row in g.iterrows():
        if pd.isna(lab):
            continue
        disc_acc[lab]['revenue'] += float(row['revenue_chunk'])
        disc_acc[lab]['qty'] += float(row['qty_chunk'])
        disc_acc[lab]['rows'] += int(row['rows_chunk'])

# Build DataFrame and save
disc_summary = pd.DataFrame([
    (lab, vals['revenue'], vals['qty'], vals['rows'])
    for lab, vals in disc_acc.items()
], columns=['disc_bin','revenue','qty','rows'])

disc_summary.to_csv(DISC_PATH, index=False)
print("Saved discount_bins_summary to:", DISC_PATH)

# Convert to crore and compute avg_margin if profit info present
disc_summary['revenue_crore'] = disc_summary['revenue'] / 1e7
disc_summary['qty_crore'] = disc_summary['qty'] / 1e7

# If a margin/profit column exists in file (it likely doesn't), calculate avg_margin; otherwise NaN
if 'profit' in disc_summary.columns:
    disc_summary['avg_margin'] = np.where(disc_summary['revenue']!=0, disc_summary['profit']/disc_summary['revenue'], np.nan)
else:
    disc_summary['avg_margin'] = np.nan

# Reorder bins to logical order (if present)
order = ['0%','0-5%','5-10%','10-20%','20-50%','50-100%','>100%']
if set(disc_summary['disc_bin'].astype(str)).issuperset(set(order)):
    disc_summary['bin_order'] = disc_summary['disc_bin'].astype(str).map({v:i for i,v in enumerate(order)})
    disc_summary = disc_summary.sort_values('bin_order').reset_index(drop=True)
else:
    disc_summary = disc_summary.sort_values('revenue', ascending=False).reset_index(drop=True)

# Display table
display_cols = ['disc_bin','revenue_crore','qty_crore','avg_margin']
print("\nDiscount bins summary (converted to Crore where applicable):")
display(disc_summary[display_cols])

# --- Plot 1: Revenue by Discount Bin (Crore ₹) ---
plt.figure(figsize=(10,4))
plt.bar(disc_summary['disc_bin'].astype(str), disc_summary['revenue_crore'])
plt.xticks(rotation=45)
plt.title("Revenue by Discount Bin (Crore ₹)")
plt.ylabel("Revenue (₹ Crore)")
plt.tight_layout()
plt.show()

# --- Plot 2: Quantity by Discount Bin (Crore units) ---
plt.figure(figsize=(10,4))
plt.bar(disc_summary['disc_bin'].astype(str), disc_summary['qty_crore'])
plt.xticks(rotation=45)
plt.title("Quantity by Discount Bin (Crore units)")
plt.ylabel("Quantity (Crore units)")
plt.tight_layout()
plt.show()

# --- Plot 3: Avg Margin by Discount Bin (if available) ---
if disc_summary['avg_margin'].notna().any():
    plt.figure(figsize=(10,4))
    plt.plot(disc_summary['disc_bin'].astype(str), disc_summary['avg_margin'], marker='o')
    plt.xticks(rotation=45)
    plt.title("Average Margin by Discount Bin")
    plt.ylabel("Avg Margin (ratio)")
    plt.tight_layout()
    plt.show()
else:
    print("No margin/profit data available in this summary to plot average margin.")

# --- Combined: Revenue (bar) + Avg Margin (line) ---
if disc_summary['avg_margin'].notna().any():
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.bar(disc_summary['disc_bin'].astype(str), disc_summary['revenue_crore'])
    ax1.set_xlabel("Discount bin")
    ax1.set_ylabel("Revenue (₹ Crore)")
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(disc_summary['disc_bin'].astype(str), disc_summary['avg_margin'], marker='o')
    ax2.set_ylabel("Avg Margin (ratio)")

    plt.title("Revenue (Crore ₹) and Avg Margin by Discount Bin")
    fig.tight_layout()
    plt.show()
else:
    print("Skipping combined revenue+margin plot (no margin data).")


# In[14]:


# 12) Pricing Anomalies Detection (Next Step After Discount Analysis)

import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
anom_path = OUT_DIR/"pricing_anomalies_sample.csv"

# Load anomaly sample created from chunked pipeline
if anom_path.exists():
    anoms = pd.read_csv(anom_path)

    # Convert selling price and landing cost to crores only if needed
    # These are per-unit values, so no crore conversion here
    print("Pricing Anomalies (sampled rows):")
    display(anoms.head(20))

    print("\nTotal anomaly rows saved:", len(anoms))

    # --- Visual: Distribution of Discount Rate for anomalies ---
    if 'discount_rate' in anoms.columns:
        plt.figure(figsize=(10,4))
        plt.hist(anoms['discount_rate'], bins=30)
        plt.title("Distribution of Discount Rate for Pricing Anomalies")
        plt.xlabel("Discount Rate")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # --- Visual: Selling Price vs Landing Cost ---
    if set(['unit_selling_price','landing_cost_per_unit']).issubset(anoms.columns):
        plt.figure(figsize=(8,5))
        plt.scatter(anoms['landing_cost_per_unit'], anoms['unit_selling_price'], alpha=0.5)
        plt.xlabel("Landing Cost per Unit")
        plt.ylabel("Selling Price per Unit")
        plt.title("Selling Price vs Landing Cost (Anomalies)")
        plt.tight_layout()
        plt.show()

else:
    print("No pricing anomalies file found. Run chunked pipeline again to generate pricing_anomalies_sample.csv.")


# In[15]:


# 13) City Clustering (Next Step After Anomalies)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")

# Load the city summary created from the chunked pipeline
city_summary = pd.read_csv(OUT_DIR/"city_summary_chunked_approx.csv")

# Convert revenue & qty to crores
city_summary['revenue_crore'] = city_summary['revenue'] / 1e7
city_summary['qty_crore'] = city_summary['total_qty'] / 1e7
city_summary['profit_crore'] = city_summary['profit'] / 1e7

# Use AOV (approx)
city_summary['AOV'] = city_summary['avg_order_value_approx']

# Only keep numeric features for clustering
cluster_df = city_summary[['revenue_crore', 'AOV', 'profit_crore']].copy()

# Drop NaN rows if any
cluster_df = cluster_df.dropna()

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(cluster_df)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
city_summary['cluster'] = kmeans.fit_predict(X)

# Show clustering result
display(city_summary[['city_name','revenue_crore','AOV','profit_crore','cluster']].head(20))

# --- Visual: Revenue vs AOV scatter (colored by cluster) ---
plt.figure(figsize=(10,6))
for c in sorted(city_summary['cluster'].unique()):
    subset = city_summary[city_summary['cluster'] == c]
    plt.scatter(subset['revenue_crore'], subset['AOV'], s=60, label=f'Cluster {c}', alpha=0.7)

plt.xlabel("Revenue (₹ Crore)")
plt.ylabel("Avg Order Value (₹)")
plt.title("City Clustering: Revenue vs AOV")
plt.legend()
plt.tight_layout()
plt.show()

# --- Visual: Profit vs Revenue (colored by cluster) ---
plt.figure(figsize=(10,6))
for c in sorted(city_summary['cluster'].unique()):
    subset = city_summary[city_summary['cluster'] == c]
    plt.scatter(subset['revenue_crore'], subset['profit_crore'], s=60, label=f'Cluster {c}', alpha=0.7)

plt.xlabel("Revenue (₹ Crore)")
plt.ylabel("Profit (₹ Crore)")
plt.title("City Clustering: Revenue vs Profit")
plt.legend()
plt.tight_layout()
plt.show()


# In[16]:


# 14) Monthly Revenue Forecasting (ARIMA) - Next Step After Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
SALES_PATH = "/Users/syedafaque/Downloads/Sales.csv"
CHUNKSIZE = 400_000  # reduce if memory issue

# Build monthly revenue using chunking
reader = pd.read_csv(
    SALES_PATH,
    usecols=['date_', 'procured_quantity', 'unit_selling_price', 'total_discount_amount'],
    parse_dates=['date_'],
    chunksize=CHUNKSIZE,
    low_memory=True
)

monthly_acc = {}

print("Aggregating monthly revenue…")
for chunk in tqdm(reader, desc="Monthly Scan"):
    # Safe numeric conversion
    chunk['procured_quantity'] = pd.to_numeric(chunk['procured_quantity'], errors='coerce').fillna(0)
    chunk['unit_selling_price'] = pd.to_numeric(chunk['unit_selling_price'], errors='coerce').fillna(0)
    chunk['total_discount_amount'] = pd.to_numeric(chunk['total_discount_amount'], errors='coerce').fillna(0)

    # Compute revenue = (price * qty) - discount
    chunk['revenue'] = chunk['unit_selling_price'] * chunk['procured_quantity'] - chunk['total_discount_amount']

    # Extract month
    chunk['month'] = chunk['date_'].dt.to_period('M')

    # Monthly aggregation for this chunk
    g = chunk.groupby('month')['revenue'].sum()

    for m, val in g.items():
        m = str(m)
        monthly_acc[m] = monthly_acc.get(m, 0) + float(val)

# Convert to Series
monthly_series = pd.Series(monthly_acc)
monthly_series.index = pd.to_datetime(monthly_series.index, format="%Y-%m")

# Save monthly revenue series
monthly_series.to_csv(OUT_DIR/"monthly_revenue_series.csv")
print("Saved monthly revenue series.")

# Convert to crores
monthly_series_crore = monthly_series / 1e7

# --- Plot monthly revenue ---
plt.figure(figsize=(12,4))
plt.plot(monthly_series_crore.index, monthly_series_crore.values)
plt.title("Monthly Revenue (₹ Crore)")
plt.ylabel("Revenue (₹ Crore)")
plt.xlabel("Month")
plt.tight_layout()
plt.show()

# --- ARIMA FORECAST ---
print("Training ARIMA model…")
model = ARIMA(monthly_series_crore, order=(1,1,1))
res = model.fit()

print(res.summary())

# 6-month forecast
forecast = res.get_forecast(steps=6)
forecast_values = forecast.predicted_mean

print("Revenue Forecast (Next 6 Months) in ₹ Crore:")
display(forecast_values)

# --- Plot forecast ---
plt.figure(figsize=(12,4))
plt.plot(monthly_series_crore.index, monthly_series_crore.values, label='History')
plt.plot(
    pd.date_range(monthly_series_crore.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq='M'),
    forecast_values,
    label='Forecast'
)
plt.title("Monthly Revenue Forecast (₹ Crore)")
plt.ylabel("Revenue (₹ Crore)")
plt.legend()
plt.tight_layout()
plt.show()


# In[17]:


# 14) Monthly Revenue Forecasting + Comparison Table (Easy to Understand)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
SALES_PATH = "/Users/syedafaque/Downloads/Sales.csv"
CHUNKSIZE = 400_000

# -----------------------------
# STEP 1: Stream monthly revenue
# -----------------------------
reader = pd.read_csv(
    SALES_PATH,
    usecols=['date_', 'procured_quantity', 'unit_selling_price', 'total_discount_amount'],
    parse_dates=['date_'],
    chunksize=CHUNKSIZE,
    low_memory=True
)

monthly_acc = {}

print("Aggregating monthly revenue…")
for chunk in tqdm(reader, desc="Monthly Scan"):
    chunk['procured_quantity'] = pd.to_numeric(chunk['procured_quantity'], errors='coerce').fillna(0)
    chunk['unit_selling_price'] = pd.to_numeric(chunk['unit_selling_price'], errors='coerce').fillna(0)
    chunk['total_discount_amount'] = pd.to_numeric(chunk['total_discount_amount'], errors='coerce').fillna(0)

    chunk['revenue'] = chunk['unit_selling_price'] * chunk['procured_quantity'] - chunk['total_discount_amount']
    chunk['month'] = chunk['date_'].dt.to_period('M')

    g = chunk.groupby('month')['revenue'].sum()

    for m, val in g.items():
        m = str(m)
        monthly_acc[m] = monthly_acc.get(m, 0) + float(val)

# Convert to Series
monthly_series = pd.Series(monthly_acc)
monthly_series.index = pd.to_datetime(monthly_series.index, format="%Y-%m")

# Convert to crores
monthly_series_crore = monthly_series / 1e7

# Save
monthly_series_crore.to_csv(OUT_DIR/"monthly_revenue_series_crore.csv")
print("Saved monthly monthly_revenue_series_crore.csv")

# -----------------------------
# STEP 2: Plot history
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(monthly_series_crore.index, monthly_series_crore.values)
plt.title("Monthly Revenue (₹ Crore)")
plt.ylabel("Revenue (₹ Crore)")
plt.xlabel("Month")
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 3: ARIMA forecasting
# -----------------------------
print("Training ARIMA…")
model = ARIMA(monthly_series_crore, order=(1,1,1))
res = model.fit()
print(res.summary())

forecast = res.get_forecast(steps=6)
forecast_values = forecast.predicted_mean

# -----------------------------
# STEP 4: Create comparison table
# -----------------------------
# Last 6 actual months
last_6_actual = monthly_series_crore.tail(6)

# Future 6 forecast months
future_months = pd.date_range(
    start=monthly_series_crore.index[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq='M'
)

forecast_df = pd.DataFrame({
    'Month': future_months,
    'Predicted_Revenue_Crore': forecast_values.values
})

actual_df = pd.DataFrame({
    'Month': last_6_actual.index,
    'Actual_Revenue_Crore': last_6_actual.values
})

comparison_table = pd.merge(
    actual_df,
    forecast_df,
    how='outer',
    on='Month'
).sort_values('Month')

print("\n📊 REVENUE COMPARISON TABLE (Actual Last 6 Months + Forecast Next 6 Months):")
display(comparison_table)

# -----------------------------
# STEP 5: Plot history + forecast
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(monthly_series_crore.index, monthly_series_crore.values, label="History")
plt.plot(future_months, forecast_values.values, label="Forecast")
plt.title("Monthly Revenue Forecast (₹ Crore)")
plt.ylabel("Revenue (₹ Crore)")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# PROFITABILITY: chunked summaries + visuals (paste into Jupyter and run)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = "/Users/syedafaque/Downloads/Sales.csv"
PRODUCTS_PATH = "/Users/syedafaque/Downloads/products.csv"
CHUNKSIZE = 400_000  # lower if you have memory issues

# --- load products master (small) ---
products = pd.read_csv(PRODUCTS_PATH, low_memory=False)
products.columns = [str(c).strip() for c in products.columns]
prod_small = products[['product_id','product_name','brand_name','l0_category','l1_category','l2_category']].copy()

# --- accumulators ---
prod_acc = {}   # product_id -> [revenue, cost, profit, qty, product_name]
brand_acc = {}  # brand -> [revenue, cost, profit, qty]
cat_acc = {}    # l1_category -> [revenue, cost, profit, qty]
city_acc = {}   # city_name -> [revenue, cost, profit, qty]

usecols = ['product_id','procured_quantity','unit_selling_price','total_discount_amount','total_weighted_landing_price','city_name']

print("Streaming Sales.csv and aggregating profitability (this may take a while)...")
reader = pd.read_csv(SALES_PATH, usecols=usecols, chunksize=CHUNKSIZE, low_memory=True)

for chunk in tqdm(reader, desc="Processing chunks"):
    # numeric safety
    chunk['procured_quantity'] = pd.to_numeric(chunk['procured_quantity'], errors='coerce').fillna(0)
    chunk['unit_selling_price'] = pd.to_numeric(chunk['unit_selling_price'], errors='coerce').fillna(0)
    chunk['total_discount_amount'] = pd.to_numeric(chunk['total_discount_amount'], errors='coerce').fillna(0)
    chunk['total_weighted_landing_price'] = pd.to_numeric(chunk['total_weighted_landing_price'], errors='coerce').fillna(0)

    # compute revenue/cost/profit per row
    chunk['revenue'] = chunk['unit_selling_price'] * chunk['procured_quantity'] - chunk['total_discount_amount']
    mask_q = chunk['procured_quantity'] > 0
    chunk['landing_cost_per_unit'] = np.nan
    if mask_q.any():
        chunk.loc[mask_q, 'landing_cost_per_unit'] = chunk.loc[mask_q, 'total_weighted_landing_price'] / chunk.loc[mask_q, 'procured_quantity']
    chunk['cost_total'] = chunk['landing_cost_per_unit'].fillna(0) * chunk['procured_quantity']
    chunk['profit'] = chunk['revenue'] - chunk['cost_total']

    # merge product metadata
    chunk = chunk.merge(prod_small, how='left', on='product_id')

    # product-level aggregation in chunk
    gprod = chunk.groupby('product_id').agg(
        revenue=('revenue','sum'),
        cost=('cost_total','sum'),
        profit=('profit','sum'),
        qty=('procured_quantity','sum')
    )
    prod_names = chunk.groupby('product_id')['product_name'].first().to_dict()
    for pid, row in gprod.iterrows():
        rev, cost, prof, qty = float(row['revenue']), float(row['cost']), float(row['profit']), float(row['qty'])
        pname = prod_names.get(pid, None)
        if pid in prod_acc:
            prod_acc[pid][0] += rev; prod_acc[pid][1] += cost; prod_acc[pid][2] += prof; prod_acc[pid][3] += qty
        else:
            prod_acc[pid] = [rev, cost, prof, qty, pname]

    # brand aggregation
    if 'brand_name' in chunk.columns:
        gbr = chunk.groupby('brand_name').agg(revenue=('revenue','sum'), cost=('cost_total','sum'), profit=('profit','sum'), qty=('procured_quantity','sum'))
        for br, row in gbr.iterrows():
            rev, cost, prof, qty = float(row['revenue']), float(row['cost']), float(row['profit']), float(row['qty'])
            if br in brand_acc:
                brand_acc[br][0] += rev; brand_acc[br][1] += cost; brand_acc[br][2] += prof; brand_acc[br][3] += qty
            else:
                brand_acc[br] = [rev, cost, prof, qty]

    # category (l1) aggregation
    if 'l1_category' in chunk.columns:
        gcat = chunk.groupby('l1_category').agg(revenue=('revenue','sum'), cost=('cost_total','sum'), profit=('profit','sum'), qty=('procured_quantity','sum'))
        for cat, row in gcat.iterrows():
            rev, cost, prof, qty = float(row['revenue']), float(row['cost']), float(row['profit']), float(row['qty'])
            if cat in cat_acc:
                cat_acc[cat][0] += rev; cat_acc[cat][1] += cost; cat_acc[cat][2] += prof; cat_acc[cat][3] += qty
            else:
                cat_acc[cat] = [rev, cost, prof, qty]

    # city aggregation
    if 'city_name' in chunk.columns:
        gcity = chunk.groupby('city_name').agg(revenue=('revenue','sum'), cost=('cost_total','sum'), profit=('profit','sum'), qty=('procured_quantity','sum'))
        for city, row in gcity.iterrows():
            rev, cost, prof, qty = float(row['revenue']), float(row['cost']), float(row['profit']), float(row['qty'])
            if city in city_acc:
                city_acc[city][0] += rev; city_acc[city][1] += cost; city_acc[city][2] += prof; city_acc[city][3] += qty
            else:
                city_acc[city] = [rev, cost, prof, qty]

# --- Build DataFrames from accumulators ---
prod_df = pd.DataFrame([(k,*v) for k,v in prod_acc.items()], columns=['product_id','revenue','cost','profit','qty','product_name']).sort_values('profit', ascending=False)
brand_df = pd.DataFrame([(k,*v) for k,v in brand_acc.items()], columns=['brand_name','revenue','cost','profit','qty']).sort_values('profit', ascending=False)
cat_df = pd.DataFrame([(k,*v) for k,v in cat_acc.items()], columns=['l1_category','revenue','cost','profit','qty']).sort_values('profit', ascending=False)
city_df = pd.DataFrame([(k,*v) for k,v in city_acc.items()], columns=['city_name','revenue','cost','profit','qty']).sort_values('profit', ascending=False)

# Profit margin & crore conversions for readability
for df in [prod_df, brand_df, cat_df, city_df]:
    df['profit_margin'] = np.where(df['revenue']==0, np.nan, df['profit']/df['revenue'])
    df['revenue_crore'] = df['revenue'] / 1e7
    df['profit_crore'] = df['profit'] / 1e7
    df['qty_crore'] = df['qty'] / 1e7

# Save CSVs for dashboards
prod_df.head(500).to_csv(OUT_DIR/"product_profitability.csv", index=False)
brand_df.to_csv(OUT_DIR/"brand_profitability.csv", index=False)
cat_df.to_csv(OUT_DIR/"category_profitability.csv", index=False)
city_df.to_csv(OUT_DIR/"city_profitability.csv", index=False)

# Display top-n tables
print("Top 20 products by profit:")
display(prod_df.head(20))
print("Top 20 brands by profit:")
display(brand_df.head(20))
print("Top 20 categories by profit:")
display(cat_df.head(20))
print("Top 20 cities by profit:")
display(city_df.head(20))

# --- VISUALS (each plot is one figure) ---

# 1) Top 10 Categories by Profit (₹ Crore)
top_cat = cat_df.head(10)
plt.figure(figsize=(10,4))
plt.bar(top_cat['l1_category'].astype(str), top_cat['profit_crore'])
plt.xticks(rotation=45)
plt.title("Top 10 Categories by Profit (₹ Crore)")
plt.ylabel("Profit (₹ Crore)")
plt.tight_layout()
plt.show()

# 2) Top 10 Brands by Profit (₹ Crore)
top_brand = brand_df.head(10)
plt.figure(figsize=(10,4))
plt.bar(top_brand['brand_name'].astype(str), top_brand['profit_crore'])
plt.xticks(rotation=45)
plt.title("Top 10 Brands by Profit (₹ Crore)")
plt.ylabel("Profit (₹ Crore)")
plt.tight_layout()
plt.show()

# 3) Top 15 Products by Profit (₹ Crore)
top_prod = prod_df.head(15)
plt.figure(figsize=(12,5))
plt.bar(top_prod['product_name'].astype(str), top_prod['profit_crore'])
plt.xticks(rotation=70)
plt.title("Top 15 Products by Profit (₹ Crore)")
plt.ylabel("Profit (₹ Crore)")
plt.tight_layout()
plt.show()

# 4) Top 15 Loss-making Products (₹ Crore)
loss_prod = prod_df[prod_df['profit'] < 0].sort_values('profit').head(15)
if not loss_prod.empty:
    plt.figure(figsize=(12,5))
    plt.bar(loss_prod['product_name'].astype(str), loss_prod['profit_crore'])
    plt.xticks(rotation=70)
    plt.title("Top 15 Loss-making Products (₹ Crore)")
    plt.ylabel("Profit (₹ Crore) [negative values]")
    plt.tight_layout()
    plt.show()
else:
    print("No loss-making products found in aggregated results.")

# 5) Profit margin distribution across products
plt.figure(figsize=(8,4))
plt.hist(prod_df['profit_margin'].dropna(), bins=40)
plt.title("Distribution of Product Profit Margins")
plt.xlabel("Profit Margin (ratio)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 6) Revenue vs Profit scatter (products) - log scale for revenue
plt.figure(figsize=(8,6))
plt.scatter(prod_df['revenue_crore'], prod_df['profit_crore'], alpha=0.6, s=20)
plt.xscale('log')
plt.xlabel("Revenue (₹ Crore, log scale)")
plt.ylabel("Profit (₹ Crore)")
plt.title("Revenue vs Profit (Products)")
plt.tight_layout()
plt.show()

print("Profitability summaries saved to:", OUT_DIR)


# In[ ]:


# ==========================================================
# FINAL VISUALISATION EXPORTER (PNG ONLY) — ROBUSTED
# Uses all CSVs generated earlier in analysis_outputs folder
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/Users/syedafaque/Downloads/analysis_outputs")
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

def safe_read_csv(p):
    p = Path(p)
    if not p.exists():
        print(f"[MISSING] {p.name}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] reading {p.name}: {e}")
        return None

# -----------------------------
# 1) CITY-WISE VISUAL (PNG)
# -----------------------------
city_path = OUT_DIR / "city_summary_chunked_approx.csv"
city_df = safe_read_csv(city_path)
if city_df is not None and not city_df.empty:
    if 'revenue' in city_df.columns:
        city_df['revenue_crore'] = pd.to_numeric(city_df['revenue'], errors='coerce') / 1e7
    else:
        city_df['revenue_crore'] = 0
    # fill city_name if missing
    if 'city_name' not in city_df.columns:
        city_df.reset_index(inplace=True)
        city_df.rename(columns={'index':'city_name'}, inplace=True)
    top10 = city_df.sort_values('revenue_crore', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(top10['city_name'].astype(str), top10['revenue_crore'])
    ax.set_xticklabels(top10['city_name'].astype(str), rotation=45, ha='right')
    ax.set_ylabel("Revenue (₹ Crore)")
    ax.set_title("Top 10 Cities by Revenue")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_top10_cities_revenue.png", dpi=150)
    plt.close(fig)
    print("Saved:", PLOT_DIR / "plot_top10_cities_revenue.png")
else:
    print("City summary CSV missing or empty; skipping city plot.")

# ---------------------------------
# 2) CATEGORY-WISE VISUAL (PNG)
# ---------------------------------
cat_path = OUT_DIR / "category_summary_chunked.csv"
cat_df = safe_read_csv(cat_path)
if cat_df is not None and not cat_df.empty:
    if 'revenue' in cat_df.columns:
        cat_df['revenue_crore'] = pd.to_numeric(cat_df['revenue'], errors='coerce') / 1e7
    else:
        cat_df['revenue_crore'] = 0
    key = 'l1_category' if 'l1_category' in cat_df.columns else (cat_df.columns[0] if len(cat_df.columns)>0 else None)
    if key is None:
        print("No category column found; skipping category plot.")
    else:
        top12 = cat_df.sort_values('revenue_crore', ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.bar(top12[key].astype(str), top12['revenue_crore'])
        ax.set_xticklabels(top12[key].astype(str), rotation=60, ha='right')
        ax.set_ylabel("Revenue (₹ Crore)")
        ax.set_title("Top 12 Categories by Revenue")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "plot_top12_categories_revenue.png", dpi=150)
        plt.close(fig)
        print("Saved:", PLOT_DIR / "plot_top12_categories_revenue.png")
else:
    print("Category summary CSV missing or empty; skipping category plot.")

# ---------------------------------
# 3) BRAND-WISE VISUAL (PNG)
# ---------------------------------
brand_path = OUT_DIR / "brand_summary_chunked.csv"
br_df = safe_read_csv(brand_path)
if br_df is not None and not br_df.empty:
    if 'revenue' in br_df.columns:
        br_df['revenue_crore'] = pd.to_numeric(br_df['revenue'], errors='coerce') / 1e7
    else:
        br_df['revenue_crore'] = 0
    name_col = 'brand_name' if 'brand_name' in br_df.columns else (br_df.columns[0] if len(br_df.columns)>0 else None)
    if name_col is None:
        print("No brand name column found; skipping brand plot.")
    else:
        top12 = br_df.sort_values('revenue_crore', ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.bar(top12[name_col].astype(str), top12['revenue_crore'])
        ax.set_xticklabels(top12[name_col].astype(str), rotation=60, ha='right')
        ax.set_ylabel("Revenue (₹ Crore)")
        ax.set_title("Top 12 Brands by Revenue")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "plot_top12_brands_revenue.png", dpi=150)
        plt.close(fig)
        print("Saved:", PLOT_DIR / "plot_top12_brands_revenue.png")
else:
    print("Brand summary CSV missing or empty; skipping brand plot.")

# ---------------------------------
# 4) TOP PRODUCTS BY PROFIT (PNG)
# ---------------------------------
prod_path = OUT_DIR / "top_products_chunked_200.csv"
prod_df = safe_read_csv(prod_path)
if prod_df is not None and not prod_df.empty:
    if 'profit' in prod_df.columns:
        prod_df['profit_crore'] = pd.to_numeric(prod_df['profit'], errors='coerce') / 1e7
    else:
        prod_df['profit_crore'] = 0
    if 'product_name' not in prod_df.columns:
        prod_df['product_name'] = prod_df['product_id'].astype(str)
    top15 = prod_df.sort_values('profit_crore', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(top15['product_name'].astype(str), top15['profit_crore'])
    ax.set_xticklabels(top15['product_name'].astype(str), rotation=75, ha='right')
    ax.set_ylabel("Profit (₹ Crore)")
    ax.set_title("Top 15 Products by Profit")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_top15_products_profit.png", dpi=150)
    plt.close(fig)
    print("Saved:", PLOT_DIR / "plot_top15_products_profit.png")
else:
    print("Top products CSV missing or empty; skipping product plot.")

# ---------------------------------
# 5) DISCOUNT BIN VISUAL (PNG)
# ---------------------------------
disc_path = OUT_DIR / "discount_bins_summary.csv"
disc_df = safe_read_csv(disc_path)
if disc_df is not None and not disc_df.empty:
    # handle different possible revenue column names
    if 'revenue' in disc_df.columns:
        rev = pd.to_numeric(disc_df['revenue'], errors='coerce') / 1e7
    elif 'revenue_crore' in disc_df.columns:
        rev = pd.to_numeric(disc_df['revenue_crore'], errors='coerce')
    else:
        # try second numeric column
        numeric_cols = [c for c in disc_df.columns if pd.api.types.is_numeric_dtype(disc_df[c])]
        rev = (pd.to_numeric(disc_df[numeric_cols[0]], errors='coerce') / 1e7) if numeric_cols else pd.Series([0]*len(disc_df))
    labels = disc_df['disc_bin'].astype(str) if 'disc_bin' in disc_df.columns else disc_df.iloc[:,0].astype(str)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(labels, rev)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Revenue (₹ Crore)")
    ax.set_title("Revenue by Discount Bin")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "plot_discount_bins_revenue.png", dpi=150)
    plt.close(fig)
    print("Saved:", PLOT_DIR / "plot_discount_bins_revenue.png")
else:
    print("Discount bins CSV missing or empty; skipping discount plot.")

# ---------------------------------
# 6) ANOMALIES – SP vs LC SCATTER (PNG)
# ---------------------------------
anom_path = OUT_DIR / "pricing_anomalies_sample.csv"
anom_df = safe_read_csv(anom_path)
if anom_df is not None and not anom_df.empty:
    if set(['unit_selling_price','landing_cost_per_unit']).issubset(anom_df.columns):
        sc = anom_df.dropna(subset=['unit_selling_price','landing_cost_per_unit'])
        if not sc.empty:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(sc['landing_cost_per_unit'].astype(float), sc['unit_selling_price'].astype(float), alpha=0.4, s=25)
            ax.set_xlabel("Landing Cost per Unit")
            ax.set_ylabel("Selling Price per Unit")
            ax.set_title("Pricing Anomalies — SP vs LC")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / "plot_anomalies_sp_vs_lc.png", dpi=150)
            plt.close(fig)
            print("Saved:", PLOT_DIR / "plot_anomalies_sp_vs_lc.png")
        else:
            print("Anomaly file has no SP/LC numeric pairs; skipping anomaly scatter.")
    else:
        print("Pricing anomalies CSV missing expected columns; skipping anomaly plot.")
else:
    print("Pricing anomalies CSV missing or empty; skipping anomaly plot.")

# ---------------------------------
# 7) MONTHLY REVENUE TREND (PNG) — robust handling of column names
# ---------------------------------
def find_time_and_value_cols(df):
    cols = df.columns.tolist()
    lower = [c.lower() for c in cols]
    # common time names
    time_candidates = ['month','date','date_','period','timestamp','unnamed: 0','index']
    time_col = None
    for tc in time_candidates:
        if tc in lower:
            time_col = cols[lower.index(tc)]
            break
    if time_col is None:
        # if first column is not numeric, treat it as time
        first = cols[0]
        if not pd.api.types.is_numeric_dtype(df[first]):
            time_col = first
    # revenue candidates
    revenue_candidates = ['revenue_crore','revenue','actual_revenue_crore','actual_revenue']
    rev_col = None
    for rc in revenue_candidates:
        if rc in lower:
            rev_col = cols[lower.index(rc)]
            break
    if rev_col is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = [c for c in numeric_cols if c != time_col]
        rev_col = numeric_cols[0] if numeric_cols else None
    return time_col, rev_col

monthly_path = OUT_DIR / "monthly_revenue_series_crore.csv"
monthly_alt = OUT_DIR / "monthly_revenue_series_raw.csv"
mdf = None
if Path(monthly_path).exists():
    mdf = pd.read_csv(monthly_path)
elif Path(monthly_alt).exists():
    mdf = pd.read_csv(monthly_alt)

if mdf is not None and not mdf.empty:
    time_col, rev_col = find_time_and_value_cols(mdf)
    if time_col is None and rev_col is None and mdf.shape[1] == 1:
        # single column; treat index as time
        mdf = mdf.reset_index().rename(columns={'index':'month', mdf.columns[0]:'revenue_crore'})
        time_col = 'month'; rev_col = 'revenue_crore'
    # parse time if available
    if time_col is not None:
        try:
            mdf['__time__'] = pd.to_datetime(mdf[time_col], errors='coerce')
            # if all NaT, fallback to original strings
            if mdf['__time__'].isna().all():
                mdf['__time__'] = mdf[time_col].astype(str)
        except:
            mdf['__time__'] = mdf[time_col].astype(str)
    else:
        mdf['__time__'] = range(len(mdf))
    if rev_col is not None:
        rev = pd.to_numeric(mdf[rev_col], errors='coerce')
        # heuristic: if values look like rupees (mean > 1e6) convert to crore
        if rev.abs().mean() > 1e6:
            mdf['revenue_crore'] = rev / 1e7
        else:
            mdf['revenue_crore'] = rev
    else:
        mdf['revenue_crore'] = np.nan

    # save standardized CSV for plotting later if needed
    mdf_out = OUT_DIR / "monthly_revenue_trend_plot_data.csv"
    mdf[['__time__','revenue_crore']].rename(columns={'__time__':'month'}).to_csv(mdf_out, index=False)
    # plot if numeric revenue exists
    if mdf['revenue_crore'].notna().any():
        fig, ax = plt.subplots(figsize=(12,5))
        x = pd.to_datetime(mdf['month'], errors='coerce') if 'month' in mdf.columns else pd.to_datetime(mdf['__time__'], errors='coerce')
        # if x all NaT, use string labels
        if pd.api.types.is_datetime64_any_dtype(x) and not x.isna().all():
            ax.plot(x, mdf['revenue_crore'], marker='o')
            ax.set_xticklabels(x.dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
        else:
            ax.plot(mdf['__time__'].astype(str), mdf['revenue_crore'], marker='o')
            ax.set_xticklabels(mdf['__time__'].astype(str), rotation=45, ha='right')
        ax.set_ylabel("Revenue (₹ Crore)")
        ax.set_title("Monthly Revenue Trend")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "plot_monthly_revenue_trend.png", dpi=150)
        plt.close(fig)
        print("Saved:", PLOT_DIR / "plot_monthly_revenue_trend.png")
    else:
        print("Monthly revenue file contained no numeric revenue; skipping monthly plot.")
else:
    print("Monthly revenue CSV not found; skipping monthly plot.")

# ---------------------------------
# 8) FORECAST PLOT (PNG) — robust handling
# ---------------------------------
fc_path = OUT_DIR / "monthly_revenue_comparison_table.csv"
fc_df = safe_read_csv(fc_path)
if fc_df is not None and not fc_df.empty:
    # try to identify month, actual, predicted columns flexibly
    cols = fc_df.columns.tolist()
    lower = [c.lower() for c in cols]
    # detect month
    month_candidates = ['month','date','date_','period','unnamed: 0']
    month_col = next((cols[i] for i,c in enumerate(lower) if c in month_candidates), None)
    # detect actual & predicted
    actual_col = next((cols[i] for i,c in enumerate(lower) if 'actual' in c), None)
    pred_col = next((cols[i] for i,c in enumerate(lower) if 'pred' in c or 'forecast' in c), None)
    # fallback heuristics
    if month_col is None:
        month_col = cols[0]
    if actual_col is None and len(cols) > 1:
        actual_col = cols[1]
    if pred_col is None and len(cols) > 2:
        pred_col = cols[2]
    # build standardized df
    try:
        plot_df = pd.DataFrame()
        plot_df['month'] = pd.to_datetime(fc_df[month_col], errors='coerce') if month_col in fc_df.columns else fc_df.index
        plot_df['actual_revenue_crore'] = pd.to_numeric(fc_df[actual_col], errors='coerce') if actual_col in fc_df.columns else np.nan
        plot_df['predicted_revenue_crore'] = pd.to_numeric(fc_df[pred_col], errors='coerce') if pred_col in fc_df.columns else np.nan
        # convert large rupee numbers
        for c in ['actual_revenue_crore','predicted_revenue_crore']:
            if plot_df[c].abs().mean(skipna=True) > 1e6:
                plot_df[c] = plot_df[c] / 1e7
        # save standardized CSV
        plot_df.to_csv(OUT_DIR / "forecast_comparison_plot_data.csv", index=False)
        # plot if at least one numeric series exists
        if plot_df[['actual_revenue_crore','predicted_revenue_crore']].notna().any().any():
            fig, ax = plt.subplots(figsize=(12,5))
            if plot_df['actual_revenue_crore'].notna().any():
                ax.plot(plot_df['month'], plot_df['actual_revenue_crore'], marker='o', label='Actual')
            if plot_df['predicted_revenue_crore'].notna().any():
                ax.plot(plot_df['month'], plot_df['predicted_revenue_crore'], marker='o', linestyle='--', label='Predicted')
            ax.legend()
            ax.set_ylabel("Revenue (₹ Crore)")
            ax.set_title("Actual vs Predicted Revenue")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / "plot_forecast_comparison.png", dpi=150)
            plt.close(fig)
            print("Saved:", PLOT_DIR / "plot_forecast_comparison.png")
        else:
            print("Forecast file had no numeric actual/predicted columns; skipping forecast plot.")
    except Exception as e:
        print("Error while plotting forecast:", e)
else:
    print("Forecast CSV not found; skipping forecast plot.")

print("\n🎉 ALL PNGS (that could be produced) are in:", PLOT_DIR)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "/Users/syedafaque/Downloads/Flipkart_Sales_Optimisation.ipynb"')


# In[ ]:




