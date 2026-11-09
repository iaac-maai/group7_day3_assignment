---
license: mit
library_name: sklearn
tags:
  - sklearn
  - clustering
  - unsupervised-learning
  - kmeans
  - built-environment
  - housing
  - iaac-maai
  - ai-methods-1
datasets:
  - iaac-maai/day3-group7-king-county-house-typologies
---

# King County house typologies: grouping houses by their physical building

This model assigns a single-family house to one of **4 house types** (post-war bungalows, mid-century houses with basements, new compact two-storey homes, large premium suburban homes) based only on the building itself: living area, plot size, floors, bedrooms, bathrooms, basement, build grade, condition and age. It is a scikit-learn pipeline (`StandardScaler` + `KMeans`, k = 4) fitted on 21,418 houses sold in King County, Washington, in 2014–15. It is meant for architects, developers or a municipality who want to describe a housing stock in recognisable types, for example to decide which type to target first in a retrofit programme. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full analysis):** https://github.com/iaac-maai/group7_day3_assignment
- **Clustered dataset:** https://huggingface.co/datasets/iaac-maai/day3-group7-king-county-house-typologies
- **Team:** Ugo, Alvaro, Sneha (Team 7)

## Intended use

- **Intended users and use:** planners, architects and students who want to place a King County single-family house in one of four building types, to describe or compare parts of the housing stock (e.g. "which share of a neighbourhood is post-war bungalows that may need retrofit").
- **Out-of-scope uses:**
  - houses outside King County or far from the 2014–15 stock (other regions, other countries, much later construction);
  - apartments, townhouse blocks, commercial or mixed-use buildings: the model only saw single-family house sales;
  - valuing a house or predicting its price: price was not an input, and the types are not price bands;
  - judging the quality of an individual property or making decisions about individual owners.

## Model details

| | |
|---|---|
| **Task** | Clustering (unsupervised) |
| **Algorithm** | K-Means |
| **Key hyperparameters** | `n_clusters=4`, `n_init=10`, `random_state=42` |
| **Preprocessing** | log10 of `sqft_living` and `sqft_lot` (done **before** the model, see *Input features*); `StandardScaler` inside the pipeline |
| **Assigner** | Not needed: K-Means has `.predict()` and assigns new houses to the nearest centroid |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline([("scale", StandardScaler()), ("kmeans", KMeans(...))])` |
| **Library versions** | Python 3.11.12, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6, joblib 1.6.0 |
| **Authors** | Ugo, Alvaro, Sneha (Team 7) |

## Input features

The model expects a `pandas.DataFrame` with exactly these 9 columns, in this order:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `log_sqft_living` | numeric | log10 sqft | Interior living area | **you compute** `log10(sqft_living)`; scaling |
| `log_sqft_lot` | numeric | log10 sqft | Plot (lot) area | **you compute** `log10(sqft_lot)`; scaling |
| `floors` | numeric | count | Number of floors (0.5 steps) | scaling |
| `bedrooms` | integer | count | Number of bedrooms | scaling |
| `bathrooms` | numeric | count | Number of bathrooms (0.25 steps) | scaling |
| `has_basement` | integer | 0/1 | 1 if the house has a basement | scaling |
| `grade` | integer | 1–13 | King County construction and design grade | scaling |
| `condition` | integer | 1–5 | King County condition rating | scaling |
| `age` | integer | years | Age at the time of observation (year − year built) | scaling |

Training ranges (min–max): living area 370–13,540 sqft, lot 520–1,651,359 sqft, floors 1–3.5, bedrooms 1–10, bathrooms 0.5–8, grade 3–13, condition 1–5, age −1–115. Inputs outside these ranges will still get a cluster, but it will not mean much.

## Output and clusters

`model.predict(X)` returns the cluster number (0–3) of each row:

| Cluster | Name | Share of the data | Defining features (medians) |
|---|---|---|---|
| 0 | mid-century houses with basements | 28.5 % | 93 % with basement, 1 floor, 2,100 sqft, 53 years old, best condition (4) |
| 1 | post-war single-storey bungalows | 29.8 % | smallest: 1,260 sqft, 1 floor, 1 bathroom, 62 years old, rarely a basement |
| 2 | large premium suburban homes | 19.0 % | largest: 3,190 sqft on 11,445 sqft plots, grade 9, 2 floors, 19 years old |
| 3 | new compact two-storey family homes | 22.7 % | newest (11 years), 2 floors, 2.5 bathrooms, smallest plots (4,725 sqft), grade 8 |

There is no noise label: every house is assigned to a cluster.

## Training data

| | |
|---|---|
| **Dataset** | House Sales in King County, USA |
| **Source** | [original](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) · [our clustered version](https://huggingface.co/datasets/iaac-maai/day3-group7-king-county-house-typologies) |
| **Licence** | CC0 1.0 (Public Domain) |
| **One row is** | One house (its most recent sale in the period) |
| **Size used** | 21,418 houses |
| **Coverage** | King County, Washington (Seattle, Bellevue, Redmond and the suburbs); sales from May 2014 to May 2015; single-family houses built 1900–2015 |

**Cleaning and feature engineering:** 18 rows with impossible values removed (Day 1/2 cleaning); for the 177 houses sold twice, only the latest sale is kept. New features: `age`, `has_basement`, and the log10 of living and lot area (both strongly right-skewed). Price, location, waterfront, view and renovation were kept aside and used only to interpret the clusters.

## Method and evaluation

- **Methods compared:** K-Means and hierarchical clustering (Ward linkage) on a random sample of 5,000 houses, both with k = 4. Agreement: **ARI = 0.63**. Three groups match almost one-to-one; Ward splits the large new houses differently.
- **Choice of k:** we tried k = 2–10. The silhouette is low and flat (0.21–0.26) and the elbow has no clear bend, so the data has no "true" k: house types are a continuum. We chose **k = 4** for stability and interpretability: k = 5 was unstable across seeds (ARI 0.59), and k = 6 adds a small but less stable "urban townhouse" group. Silhouette at k = 4: **0.23**.
- **Stability:** re-running with 5 other random seeds gives mean **ARI 0.997**; refitting on 5 random 80 % subsamples gives mean **ARI 0.993**. The same four types come back every time.
- **Interpretation check:** although location and price were not inputs, the types separate by place (the two older types in Seattle, the two newer in the suburbs) and by price (median $334K for bungalows to $771K for premium homes).
- **Why this model:** K-Means is stable here, its groups are nameable, and it can assign new houses directly with `.predict()`.

## Limitations and biases

- **Weak separation:** with a silhouette of 0.23, many houses lie between two types. A prediction for a house near a border is much less meaningful than for a typical member. Check the distance to the centroids (`model.transform(X)`) if this matters.
- **One reading among several:** the groups depend on our feature choice and on k. With other features (e.g. price or location) you would find markets or neighbourhoods, not building types.
- **Sold houses only:** the training data contains houses sold in 2014–15 in one county. Unsold houses, apartments and other building types are not represented, and newer or older construction styles may be misassigned.
- **Age drifts:** `age` was measured at sale in 2014–15. A house observed today should use today's age, but the training distribution is centred on that period.
- **Assessor ratings:** `grade` and `condition` are King County assessor scales. Values from another rating system are not comparable.

## How to use

```python
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day3-group7-house-typologies", filename="model.joblib")
model = joblib.load(path)

CLUSTER_NAMES = {
    0: "mid-century houses with basements",
    1: "post-war single-storey bungalows",
    2: "large premium suburban homes",
    3: "new compact two-storey family homes",
}

house = {"sqft_living": 1800, "sqft_lot": 4000, "floors": 2, "bedrooms": 3, "bathrooms": 2.5,
         "has_basement": 0, "grade": 8, "condition": 3, "age": 5}

example = pd.DataFrame([{
    "log_sqft_living": np.log10(house["sqft_living"]),
    "log_sqft_lot": np.log10(house["sqft_lot"]),
    "floors": house["floors"],
    "bedrooms": house["bedrooms"],
    "bathrooms": house["bathrooms"],
    "has_basement": house["has_basement"],
    "grade": house["grade"],
    "condition": house["condition"],
    "age": house["age"],
}])

cluster = model.predict(example)[0]
print(cluster, CLUSTER_NAMES[cluster])   # -> 3 new compact two-storey family homes
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above (especially scikit-learn 1.9.1); the model may not load with other versions.

## Reproducibility

The full analysis (data loading, feature engineering, PCA and UMAP, clustering, stability, interpretation) is in the GitHub repository: https://github.com/iaac-maai/group7_day3_assignment. Run `notebook/unsupervised_learning.ipynb` with the packages in `requirements.txt`. All randomness uses `random_state=42`.

## Authors and licence

Ugo, Alvaro and Sneha (Team 7). Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction), released under CC0 1.0.
