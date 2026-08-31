# E-commerce Customer Segmentation

Customer segmentation on the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (~1.07M transactions, UK-based online retailer, Dec 2009–Dec 2011), comparing a centroid-based, a density-based, and a hierarchical clustering algorithm head-to-head, then turning the result into customer personas with recommended actions.

## Pipeline

| Notebook | What it does |
|---|---|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Raw-data profiling: missingness, invoice/StockCode anomalies, seasonality, revenue concentration |
| [`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb) | Cancellations, non-positive qty/price, missing `Customer_ID`, non-product codes, dedup |
| [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | Per-customer RFM + basket size, category diversity, international flag; log-transform + standardize |
| [`03_clustering_comparison.ipynb`](notebooks/03_clustering_comparison.ipynb) | K-Means, DBSCAN, Agglomerative on the same feature set; parameter selection + evaluation |
| [`04_personas_and_viz.ipynb`](notebooks/04_personas_and_viz.ipynb) | PCA/UMAP visualization; final hybrid segmentation and persona writeup |

## Data cleaning

Started at 1,067,371 raw rows, 22.8% missing `Customer_ID`. Cleaning steps (in order — order matters, since dropping missing-`Customer_ID` rows first eliminates most of the junk `StockCode`s automatically):

1. Drop cancelled invoices (`Invoice` starts with "C")
2. Drop non-positive `Quantity`/`Price` (confirmed via `Description` to be returns/losses/adjustments, not sales)
3. Drop rows with missing `Customer_ID`
4. Drop remaining non-product `StockCode`s — `POST`, `M`, `C2`, `ADJUST`, `ADJUST2`, `BANK CHARGES`, `DOT`, `TEST001`, `TEST002`, `D` — decided by inspecting each code's `Description`, not a blanket regex. Codes like `15056BL` (Edwardian Parasol) and `79323LP`/`79323GR` (cherry lights) are real products with non-standard codes and were intentionally kept.
5. Drop exact duplicate rows

**Limitation**: ~23% of transactions have no `Customer_ID` and are excluded entirely — segmentation only covers identifiable customers, and December 2011 is a partial month (data cuts off on the 9th), which affects any month-over-month comparison involving it.

## Features

Recency, Frequency, Monetary (RFM) computed at the invoice level, plus:
- **Average basket size** — mean items per invoice
- **Category diversity** — distinct `StockCode`s purchased
- **International flag** — shipped outside the UK at least once

`Monetary` and `Frequency` are heavily right-skewed (median spend £875 vs. a top customer at £608,821) and were log1p-transformed before standardizing all six features.

## Algorithm comparison

All three run on the identical scaled feature set (5,852 customers × 6 features):

| Algorithm | Parameters | Clusters | Outliers flagged | Silhouette | Davies-Bouldin |
|---|---|---|---|---|---|
| K-Means | k=3 (elbow + silhouette) | 3 | No — every point assigned | 0.356 | — |
| DBSCAN | eps=0.7, min_samples=5 (k-distance plot) | 6 | Yes — 190 customers (3.2%) | 0.337 (excl. noise) | — |
| Agglomerative (ward) | k=3 (dendrogram) | 3 | No | **0.375** | 1.048 |

**Why not just pick the highest silhouette score:** Agglomerative technically scores best, and K-Means agrees closely on the same 3-cluster macro-structure — that agreement across two independent algorithms is a good sanity check on the segmentation. But neither can flag outliers; every customer gets forced into one of three buckets. Checking K-Means' "high value" cluster against DBSCAN's noise flag showed that **108 of DBSCAN's 190 flagged outliers were sitting inside K-Means' single high-value cluster**, quietly pulling its average up without being distinguishable from a merely-good customer. That's the concrete version of the "K-Means forces a whale into a cluster" problem this project set out to demonstrate — DBSCAN's silhouette is lower, but it's answering a question the other two can't: *which customers are structurally unlike everyone else, not just at the high end of a smooth range.*

## Final segmentation: hybrid

K-Means' 3-cluster base, with DBSCAN's outlier flag layered on top as a 4th, higher-priority segment (pulled out first, before applying the K-Means label to what's left):

| Segment | Customers | Avg recency | Avg frequency | Avg spend | % international | % of revenue |
|---|---|---|---|---|---|---|
| **VIP Outliers** | 190 (3.2%) | 206 days | 25.4 | £27,002 | 34% | **29.5%** |
| **Active High-Value Core** | 1,930 (33%) | 56 days | 11.8 | £4,910 | 0% | 54.5% |
| **Lapsed Low-Value** | 3,279 (56%) | 288 days | 2.1 | £564 | 0% | 10.6% |
| **International** | 453 (8%) | 175 days | 4.3 | £2,038 | 100% | 5.3% |

### Personas

**1. VIP Outliers** — 190 customers, 3.2% of the base, **29.5% of revenue**. DBSCAN-flagged density outliers: too large or unusual to belong to any "normal" cluster, and not always the most recent buyers (avg recency 206 days, higher than the active core). *Action: dedicated account management / personal outreach — losing even one of these customers has an outsized revenue impact that a generic campaign won't protect against.*

**2. Active High-Value Core** — 1,930 customers, the main revenue engine at 54.5% of revenue. Recent (56 days), frequent (11.8 orders), reliable. *Action: loyalty/rewards program to retain this segment and create a path toward VIP status.*

**3. Lapsed Low-Value** — 3,279 customers, the largest group by count but smallest by revenue (10.6%). Long recency (288 days), low frequency (2.1 orders). *Action: low-cost automated win-back email with a discount incentive; don't over-invest given historically low value.*

**4. International** — 453 customers, 100% international, 5.3% of revenue. Meaningfully lower purchase frequency (4.3) than the domestic active core despite comparable recency. *Action: investigate shipping cost/time or currency friction as a likely driver before assuming lower loyalty; consider localized campaigns or region-specific offers.*

## Limitations

- ~23% of transactions have no `Customer_ID` and are excluded from segmentation entirely.
- December 2011 is a partial month (data ends Dec 9), which understates that month in any time-series view.
- PCA captures only ~66% of variance in 2 components — the 2D scatter plots (PCA and UMAP, in notebook 04) are illustrative, not a complete picture of the 6-dimensional feature space.
- RFM + the extra features used here don't include product-category-level purchase patterns (what customers buy, not just how often/how much) — a natural extension for a recommendation-style follow-up.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=ecommerce-seg --display-name "Ecommerce Segmentation"
```

Run the notebooks in order (01 → 04); each stage saves its output to `data/processed/` for the next.
