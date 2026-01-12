import arviz

print(arviz.__version__)
print(arviz.__file__)

# Load required libraries and custom functions
import os

from Functions.ml_models import *
from Functions.dl_models import *
from Functions.cv_functions import *
from Functions.nlp_functions import *
from Functions.general_functions import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


import json
import pandas as pd

def prepare_dataset(df, feature, include_poi):
    """
    Creates a clean modeling dataset with:
    - Always includes price
    - Optionally includes LLM feature
    - Optionally includes POI column
    - Removes ID columns & unused LLM features
    """

    # Always remove raw LLM features except the one we want
    drop_cols = ['n_specific', 'n_total', 'n_generalParent', 'listing_id']

    # Start with all usable columns
    use_cols = [c for c in df.columns if c not in drop_cols]

    # ---- Add LLM feature only if requested ----
    if feature != "none":
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found in dataframe!")
        use_cols.append(feature)

    # ---- Ensure 'price' is included ----
    if "price" not in use_cols:
        use_cols.append("price")

    # ---- Handle POI ----
    if not include_poi:
        if "poi_within_800m" in use_cols:
            use_cols.remove("poi_within_800m")

    # Final dataset
    df_model = df[use_cols].copy()

    # Basic safety check
    if "price" not in df_model.columns:
        raise RuntimeError("Critical error: price column missing after filtering!")

    return df_model

def run_xgb(df_model, feature_name):
    """
    Runs imputation + XGBoost model on a prepared dataset.
    df_model must already contain a valid set of features + price.
    """

    # -----------------------------
    # Train-test split
    # -----------------------------
    y = df_model["price"]
    X = df_model.drop(columns=["price"])

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.09, random_state=123
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.15, random_state=123
    )

    X_train, X_valid, X_test = new_id(X_train), new_id(X_valid), new_id(X_test)
    y_train, y_valid, y_test = new_id(y_train), new_id(y_valid), new_id(y_test)

    # -----------------------------
    # Imputation selection
    # -----------------------------
    rng = default_rng(seed=42)
    random_cols = rng.choice(X_test.columns, size=min(10, len(X_test.columns)), replace=False)
    random_rows = rng.choice(X_test.index, size=max(1, int(0.10 * len(X_test))), replace=False)

    knn_imp = imputer_performance((X_test, y_test), "knn", random_cols, random_rows)
    bayes_imp = imputer_performance((X_test, y_test), "bayes", random_cols, random_rows)
    forest_imp = imputer_performance((X_test, y_test), "forest", random_cols, random_rows)

    # Determine best imputer
    if (forest_imp < knn_imp).sum() > 0 and (forest_imp < bayes_imp).sum() > 0:
        best = "forest"
    elif (knn_imp < bayes_imp).sum() > (forest_imp < bayes_imp).sum():
        best = "knn"
    else:
        best = "bayes"

    # Apply the best imputer
    if best == "forest":
        X_train, _ = Iterative_Imputer(X_train, y_train, model="forest")
        X_valid, _ = Iterative_Imputer(X_valid, y_valid, model="forest")
        X_test, _ = Iterative_Imputer(X_test, y_test, model="forest")
    elif best == "knn":
        X_train = KNN_Imputer(X_train, y_train, k=6)
        X_valid = KNN_Imputer(X_valid, y_valid, k=6)
        X_test = KNN_Imputer(X_test, y_test, k=6)
    else:
        X_train, _ = Iterative_Imputer(X_train, y_train, model="bayesian")
        X_valid, _ = Iterative_Imputer(X_valid, y_valid, model="bayesian")
        X_test, _ = Iterative_Imputer(X_test, y_test, model="bayesian")

    # Final forest consistency pass
    X_train, _ = Iterative_Imputer(X_train, y_train, model="forest")
    X_valid, _ = Iterative_Imputer(X_valid, y_valid, model="forest")
    X_test, _ = Iterative_Imputer(X_test, y_test, model="forest")

    # Encode categorical
    categorical = ['property', 'room_type', 'neighborhood_group',
                   'host_response_time', 'gender_final']

    for c in categorical:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_valid[c] = X_valid[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    # Fit final model
    train = [X_train, y_train]
    valid = [X_valid, y_valid]
    test = [X_test, y_test]

    model, rmse, hyper, _, _ = XGBoost(train, valid, test)

    return rmse, hyper


# -----------------------------
# PARAMETERS
# -----------------------------
feature_list = ['none']#, 'n_specific', 'n_total', 'n_generalParent']
poi_options = [False, True]
geojson_options = [False, True]

# -----------------------------
# Load Base Listings Once
# -----------------------------
base_listings = pd.read_csv(
    r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\tuneXgboost_feature4.csv",
    index_col=0
)

# Load geojson filter IDs
with open(r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\listings_home.geojson", "r") as f:
    geo_ids = json.load(f)

results = []

# =============================================================
#                  LOOP THROUGH CASES
# =============================================================
for feature in feature_list:
    for geo_filter in geojson_options:
        for include_poi in poi_options:

            df = base_listings.copy()

            # Apply geojson filter if needed
            if geo_filter:
                df = df[df["listing_id"].isin(geo_ids)]

            # Build clean modeling dataset
            df_model = prepare_dataset(df, feature, include_poi)

            print(f"Running → feature={feature}, geo={geo_filter}, poi={include_poi}")

            rmse, hyper = run_xgb(df_model, feature)

            results.append({
                "feature": feature,
                "geojson_filter": geo_filter,
                "poi_within_800m": include_poi,
                "rmse": rmse,
                "hyper": hyper
            })




            print(f"✓ Completed → RMSE={rmse:.4f}\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(
    r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\xgb_llm_feature_comparison_full_Baseline.csv",
    index=False
)

print("===== RESULTS =====")
print(results_df)