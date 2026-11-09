---
license: cc0-1.0
pretty_name: King County House Typologies
size_categories:
  - 10K<n<100K
tags:
  - built-environment
  - housing
  - clustering
  - unsupervised-learning
  - iaac-maai
  - ai-methods-1
configs:
  - config_name: default
    data_files: king_county_house_typologies.csv
---

# King County House Typologies

21,418 single-family houses sold in King County, Washington (Seattle and its suburbs) between May 2014 and May 2015. Each house is assigned to one of four **house types** found by clustering only the physical building: size, plot, floors, rooms, basement, build quality, condition and age. Price and location were deliberately left out of the clustering and are kept in the table to show how the types differ in value and place. **One row = one house** (its most recent sale in the period).

- **GitHub repository (how it was made):** https://github.com/iaac-maai/group7_day3_assignment
- **Model fitted on it:** https://huggingface.co/iaac-maai/day3-group7-house-typologies
- **Team:** Ugo, Alvaro, Sneha (Team 7)

## Source and attribution

| | |
|---|---|
| **Original dataset** | [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (Kaggle, uploaded by harlfoxem), based on King County public sales records |
| **Original licence** | CC0 1.0 (Public Domain) |

## What we changed

- **Cleaning (Day 1/2):** 18 rows with impossible values removed from the original 21,613 sales. Types made consistent (`date` as a date, `zipcode` as text).
- **Deduplication:** 177 houses were sold twice in the period. We keep only the most recent sale, so each house appears once (21,595 → 21,418 rows).
- **Privacy:** the original `id` (King County parcel id) is dropped. See *Limitations* about the coordinates.
- **Feature engineering (8 new columns):** `age`, `renovated`, `has_basement`, `sqft_per_bedroom`, `floor_area_ratio`, and the log10 of living area, lot size and floor area ratio (these are strongly right-skewed: a few huge plots would otherwise dominate the distances).
- **Clustering:** `cluster` and `cluster_name` come from the K-Means model linked above (k = 4, on 9 standardised building features). The coordinates come from PCA (first two components) and UMAP (`n_neighbors=30`, `min_dist=0.1`, `random_state=42`), both on the same standardised features.

## The clusters

![The house types on the map of King County](https://huggingface.co/datasets/iaac-maai/day3-group7-king-county-house-typologies/resolve/main/clusters_map.png)

| Cluster | Name | Share of the rows | What makes it different |
|---|---|---|---|
| 0 | mid-century houses with basements | 28.5 % (6,095) | 93 % have a basement, 1 floor above ground, ~2,100 sqft, median age 53, best condition (median 4/5). Inner Seattle (98115, 98117). Median price $497K |
| 1 | post-war single-storey bungalows | 29.8 % (6,388) | Smallest (1,260 sqft), 1 floor, 1 bathroom, median age 62, rarely a basement. Central Seattle (98118, 98117, 98133). Cheapest ($334K) but the highest price per sqft ($271) |
| 2 | large premium suburban homes | 19.0 % (4,080) | Largest (3,190 sqft) on large plots (11,445 sqft), grade 9, 2 floors, ~19 years old. Eastside: Sammamish, Bellevue, Redmond (98075, 98074, 98006). Most expensive ($771K), most often with a view or on the water |
| 3 | new compact two-storey family homes | 22.7 % (4,855) | Newest (median 11 years), 2 floors, 2.5 bathrooms, smallest plots (4,725 sqft). Suburban developments in the south and east (98038, 98029, 98042). $415K, lowest price per sqft ($227) |

**Reading.** The four types are **very stable** (adjusted Rand index 0.997 across random seeds and 0.993 across 80 % subsamples) but **not sharply separated** (silhouette 0.23, flat for every k from 2 to 10): house types form a continuum and many houses sit near a border. Hierarchical clustering (Ward) partly agrees with K-Means (ARI 0.63). Although location was never used, the types separate by place: the two older types are in Seattle itself, the two newer types in the suburbs. Price follows size and quality more than type: the bungalows are the cheapest houses but cost the most per square foot, because of where they are.

## Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| `date` | date | – | Date of sale (2014-05-02 to 2015-05-27) |
| `price` | float | USD | Sale price |
| `bedrooms` | integer | count | Number of bedrooms |
| `bathrooms` | float | count | Number of bathrooms (0.5 = toilet without shower, 0.75 = with shower) |
| `sqft_living` | integer | sqft | Interior living area |
| `sqft_lot` | integer | sqft | Plot (lot) area |
| `floors` | float | count | Number of floors (0.5 = half floor) |
| `waterfront` | integer | 0/1 | 1 if the house overlooks the waterfront |
| `view` | integer | 0–4 | Quality of the view |
| `condition` | integer | 1–5 | Upkeep of the house (King County scale) |
| `grade` | integer | 1–13 | Construction and design quality (King County scale) |
| `sqft_above` | integer | sqft | Living area above ground |
| `sqft_basement` | integer | sqft | Living area in the basement |
| `yr_built` | integer | year | Year built |
| `yr_renovated` | integer | year | Year of last renovation (0 = never) |
| `zipcode` | text | – | ZIP code |
| `lat`, `long` | float | degrees | Latitude and longitude of the house |
| `sqft_living15` | integer | sqft | Average living area of the 15 nearest neighbours |
| `sqft_lot15` | integer | sqft | Average lot area of the 15 nearest neighbours |
| `age` | integer | years | Age at sale: sale year − `yr_built` (−1 = sold before completion) |
| `renovated` | integer | 0/1 | 1 if `yr_renovated > 0` |
| `has_basement` | integer | 0/1 | 1 if `sqft_basement > 0` |
| `sqft_per_bedroom` | float | sqft | `sqft_living / bedrooms` |
| `floor_area_ratio` | float | – | `sqft_living / sqft_lot`: how much of the plot the house uses |
| `log_sqft_living` | float | log10 sqft | log10 of `sqft_living` (clustering feature) |
| `log_sqft_lot` | float | log10 sqft | log10 of `sqft_lot` (clustering feature) |
| `log_floor_area_ratio` | float | – | log10 of `floor_area_ratio` |
| `cluster` | integer | – | Cluster number (0–3) assigned by the K-Means model |
| `cluster_name` | text | – | Name of the cluster (our interpretation, see table above) |
| `pca_x`, `pca_y` | float | – | Position on the first two principal components (39 % and 19 % of the variance) of the 9 standardised clustering features |
| `umap_x`, `umap_y` | float | – | Position on the UMAP map (`n_neighbors=30`, `min_dist=0.1`, `random_state=42`); only for plotting. Distances between groups and blob sizes are not meaningful |

The 9 features used for clustering are `log_sqft_living`, `log_sqft_lot`, `floors`, `bedrooms`, `bathrooms`, `has_basement`, `grade`, `condition` and `age`. The other columns were kept aside for interpretation.

## Limitations and biases

- **Coverage:** only houses that **sold** in one US county in 2014–15. Houses that did not sell, apartments and non-residential buildings are missing, so the types describe the market of that year, not the whole building stock. Do not apply them to other regions or periods.
- **One possible reading:** the clusters depend on our choice of features and of k. Other features or another k give other groups (at k = 6 a small "urban townhouse" group on tiny plots appears). The names are our interpretation of the profiles, not ground truth.
- **Fuzzy borders:** the silhouette is low (0.23). Many houses sit between two types, and their label means less than that of a typical member.
- **Not a valuation:** the types describe buildings, not their value or quality. Do not use `cluster_name` to price or judge an individual property.
- **Location and privacy:** the table keeps the exact `lat` / `long`, sale date and price of each house, as in the public original. These come from public sales records and contain no names, but combined they can point to a specific property. Use them for aggregate analysis only.
- **King County scales:** `grade`, `condition` and `view` are assessor ratings, with the subjectivity that implies.

## How to load it

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hf://datasets/iaac-maai/day3-group7-king-county-house-typologies/king_county_house_typologies.csv")

# the clusters on the UMAP map (use pca_x / pca_y, or long / lat, for the other maps)
for name, rows in df.groupby("cluster_name"):
    plt.scatter(rows["umap_x"], rows["umap_y"], s=2, alpha=0.5, label=name)
plt.legend(markerscale=6)
plt.show()
```

## Authors and licence

Ugo, Alvaro and Sneha (Team 7). Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Released under **CC0 1.0**, based on [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (CC0 1.0).
