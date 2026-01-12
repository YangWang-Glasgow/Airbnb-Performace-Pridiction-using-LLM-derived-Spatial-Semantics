import arviz
import os
import json
import shap
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from numpy.random import default_rng


from Functions.ml_models import *
from Functions.general_functions import *

from joblib import dump, load


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


sns.set_theme()   # avoids seaborn future warnings





# ======================================================
#   CLEAN DATA PREPARATION FUNCTION
# ======================================================
def prepare_dataset(df, feature, include_poi):
    """
    Build modeling dataset with flexible choices:
    - feature = 'none' → no LLM feature included
    - include_poi = True/False
    """
    print(df.columns)
    print(df.shape)
    drop_cols = ['listing_id', 'geo_z','geo_y','geo_x','calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms',
                 'property_type','neighborhood_group', 'has_availability','availability_365','availability_90', 'no_review','source',
                 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'source', 'description', 'geometry', 'buffer_800m',
                 'n_specific', 'n_total', 'n_generalParent']

    # Start with all base columns except always-dropped ones
    use_cols = [c for c in df.columns if c not in drop_cols]

    # Add LLM feature ONLY when requested
    if feature != "none":
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found in dataframe!")
        if feature not in use_cols:
            use_cols.append(feature)

    # Ensure price is present
    if "price" not in use_cols:
        use_cols.append("price")

    # Remove POI variable if user disables it
    if not include_poi and "poi_within_800m" in use_cols:
        use_cols.remove("poi_within_800m")

    df_model = df[use_cols].copy()

    if "price" not in df_model.columns:
        raise RuntimeError("FATAL: price missing after dataset filtering!")

    return df_model


def make_dirs(base_dir, prefix):
    dirs = {
        # "root": f"{base_dir}/{prefix}",
        # "xgb": f"{base_dir}/{prefix}/xgb_importance",
        # "shap_global": f"{base_dir}/{prefix}/shap_global",
        # "shap_dependence": f"{base_dir}/{prefix}/shap_dependence",
        # "shap_local": f"{base_dir}/{prefix}/shap_local",
        # "lime": f"{base_dir}/{prefix}/lime",
        # "pdp_ice": f"{base_dir}/{prefix}/pdp_ice",
        # "rmse": f"{base_dir}/rmse_comparisons",

        "interaction": f"{base_dir}/{prefix}/shap_interations",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs




def run_rf_full(df_model, save_prefix, base_dir):
    """
    Full pipeline for:
    - train/valid/test split
    - imputation model selection
    - Random Forest training via freq_random_forest()
    - feature importance (RF built-in + seaborn)
    - SHAP (TreeExplainer)
    - SHAP dependence
    - SHAP local waterfall
    - LIME explanation
    - PDP + ICE

    Returns:
        rmse_test, df_importance
    """

    dirs = make_dirs(base_dir, save_prefix)

    # ---------------------------
    # Split data
    # ---------------------------
    y = df_model["price"]
    X = df_model.drop(columns=["price"])

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.10, random_state=123
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.15, random_state=123
    )

    X_train, X_valid, X_test = map(new_id, [X_train, X_valid, X_test])
    y_train, y_valid, y_test = map(new_id, [y_train, y_valid, y_test])

    # ---------------------------
    # Missing-data imputation
    # ---------------------------
    rng = default_rng(42)
    random_cols = rng.choice(X_test.columns, min(10, len(X_test.columns)), replace=False)
    random_rows = rng.choice(X_test.index, int(0.1 * len(X_test)), replace=False)

    knn_imp = imputer_performance((X_test, y_test), "knn", random_cols, random_rows)
    bayes_imp = imputer_performance((X_test, y_test), "bayes", random_cols, random_rows)
    forest_imp = imputer_performance((X_test, y_test), "forest", random_cols, random_rows)

    if (forest_imp < knn_imp).sum() > 0 and (forest_imp < bayes_imp).sum() > 0:
        best = "forest"
    elif (knn_imp < bayes_imp).sum() > (forest_imp < bayes_imp).sum():
        best = "knn"
    else:
        best = "bayes"

    # apply the best imputer
    if best == "forest":
        X_train, _ = Iterative_Imputer(X_train, y_train, model="forest")
        X_valid, _ = Iterative_Imputer(X_valid, y_valid, model="forest")
        X_test,  _ = Iterative_Imputer(X_test,  y_test,  model="forest")
    elif best == "knn":
        X_train = KNN_Imputer(X_train, y_train, k=6)
        X_valid = KNN_Imputer(X_valid, y_valid, k=6)
        X_test  = KNN_Imputer(X_test,  y_test,  k=6)
    else:
        X_train, _ = Iterative_Imputer(X_train, y_train, model="bayesian")
        X_valid, _ = Iterative_Imputer(X_valid, y_valid, model="bayesian")
        X_test,  _ = Iterative_Imputer(X_test,  y_test,  model="bayesian")

    # ---------------------------
    # Encode categorical variables
    # ---------------------------
    categorical = ['property', 'room_type', 'host_response_time', 'gender_final']
    for col in categorical:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_valid[col] = X_valid[col].astype("category").cat.codes
            X_test[col]  = X_test[col].astype("category").cat.codes

    # ---------------------------
    # Train Random Forest (YOUR FUNCTION)
    # ---------------------------
    rmse_test, best_params, n_features, feature_names = freq_random_forest(
        [X_train, y_train],
        [X_valid, y_valid],
        [X_test,  y_test],
    )



    # Fit final RF again to extract feature importance and SHAP
    rf_model = RandomForestRegressor(**best_params, random_state=123, n_jobs=-1)


    X_final = pd.concat([X_train, X_valid])
    y_final = pd.concat([y_train, y_valid])
    rf_model.fit(X_final, y_final)

    preds = rf_model.predict(X_test)

    # ---------------------------
    # Feature Importance
    # ---------------------------
    importances = rf_model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)


    # # SHAP dependence (top 5)
    # top_features = df_importance.head(5)["feature"].tolist()
    #
    # ---------------------------
    # SHAP interaction values
    # ---------------------------
    explainer = shap.TreeExplainer(rf_model)
    shap_inter = explainer.shap_interaction_values(X_test)

    # Advanced SHAP interaction for your two key variables
    plot_advanced_shap_interaction_pair(
        primary_feature_name="n_total",
        interacting_feature_name="poi_within_800m",
        shap_interaction_values=shap_inter,
        X_test=X_test,
        save_folder=f"{base_dir}/{prefix}/shap_interations"
    )


    return rmse_test, df_importance



base_dir = r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\rf_experiments_22"

feature_list = ['n_total']#'none','n_total', 'n_generalParent']#'n_specific', 'n_total'
poi_options = [ True]#False,
geojson_options = [False]#, True

base_listings = pd.read_csv(
    r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\ModelFitting_clean_202209.csv",
    index_col=0
)
with open(r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\listings_home.geojson") as f:
    geo_ids = json.load(f)

results = []
shap_tables = {}

for feature in feature_list:
    for geo in geojson_options:
        for poi in poi_options:

            df = base_listings.copy()
            if geo:
                df = df[df["listing_id"].isin(geo_ids)]

            df_model = prepare_dataset(df, feature, poi)

            prefix = f"{feature}_geo{geo}_poi{poi}_interaction"
            print("Running →", prefix)

            rmse, df_gain = run_rf_full(df_model, prefix, base_dir)

            shap_tables[prefix] = df_gain

            results.append({
                "feature": feature,
                "geo": geo,
                "poi": poi,
                "rmse": rmse
            })

            results_df = pd.DataFrame(results)
            # results_df.to_csv(f"{base_dir}/rmse_comparisons/experiment_results_{prefix}.csv", index=False)
            print(results_df)


