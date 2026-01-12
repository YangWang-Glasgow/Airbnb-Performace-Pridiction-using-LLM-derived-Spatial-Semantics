
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


# ======================================================
#   CLEAN DATA PREPARATION FUNCTION
# ======================================================
def prepare_dataset(df, feature, include_poi):
    """
    Build modeling dataset with flexible choices:
    - feature = 'none' → no LLM feature included
    - include_poi = True/False
    """

    drop_cols = ['listing_id',
                 'price',#'host_listings_count',
                 'first_review','last_review','reviews_month',
                 'n_specific', 'n_total', 'n_generalParent']


    # Start with all base columns except always-dropped ones
    use_cols = [c for c in df.columns if c not in drop_cols]

    # Add LLM feature ONLY when requested
    if feature != "none":
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found in dataframe!")
        if feature not in use_cols:
            use_cols.append(feature)

    # # Ensure price is present
    # if "price" not in use_cols:
    #     use_cols.append("price")


    # Remove POI variable if user disables it
    if not include_poi and "poi_within_800m" in use_cols:
        use_cols.remove("poi_within_800m")

    df_model = df[use_cols].copy()

    if "number_of_reviews" not in df_model.columns:
        raise RuntimeError("FATAL: number_of_reviews missing after dataset filtering!")



    return df_model

def run_full_model_selection(df_model, geo_filter):
    """
    Runs model selection (Bayesian Ridge, Elastic Net, RF, ANN, BNN, XGB)
    only for baseline feature='none' and include_poi=False.
    """

    # -------------------------------
    # Extract target + features
    # -------------------------------
    y = df_model['number_of_reviews']
    X = df_model.drop(columns=['number_of_reviews'])

    # train/valid/test split
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.09, random_state=123)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.15, random_state=123)

    # Reset index
    X_train, X_valid, X_test = new_id(X_train), new_id(X_valid), new_id(X_test)
    y_train, y_valid, y_test = new_id(y_train), new_id(y_valid), new_id(y_test)

    # Stack features and target
    train = [X_train, y_train]
    valid = [X_valid, y_valid]
    test = [X_test, y_test]

    # =======================
    # 1) Imputation Selection
    # =======================
    rng = default_rng(42)
    random_cols = rng.choice(X_test.columns, 10, replace=False)
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

    # Apply best imputer
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

    ###categoricalized done when data generating
    # # Categorical → codes
    # categorical = ['property', 'room_type', #'neighborhood_group',
    #                'host_response_time', 'gender_final']
    # for col in categorical:
    #     if col in X_train.columns:
    #         X_train[col] = X_train[col].astype("category").cat.codes
    #         X_valid[col] = X_valid[col].astype("category").cat.codes
    #         X_test[col] = X_test[col].astype("category").cat.codes
    #
    # categorical = [
    #     'property', 'room_type', #'neighborhood_group',
    #     'host_response_time', 'gender_final'
    # ]
    # # Correctly encode categorical features
    # for i in categorical:
    #     X_train[i] = X_train[i].astype('category')
    #     X_valid[i] = X_valid[i].astype('category')
    #     X_test[i] = X_test[i].astype('category')
    ###categoricalized done when data generating


    # Stack final features and target
    train = [X_train, y_train]
    valid = [X_valid, y_valid]
    test = [X_test, y_test]

    print('trainig start --------------')
    model_results = []

    # Retrieve results from Bayesian Ridge Regression
    print("Training bayesian_regression..")

    rmse_bayes, hyperparams, intercept, coefs = bayesian_regression(train, valid, test)
    print('---------------')
    print(rmse_bayes)
    print('---------------')
    print(hyperparams)
    print('---------------')
    print(intercept)
    print('---------------')
    print(coefs)
    model_results.append({
        "model": "Bayesian Ridge Regression",
        "rmse": rmse_bayes,
    })

    # Retrieve results from Elastic Net Regularized Linear Regression
    print("Training elastic_net_OLS..")

    rmse_enet, hyperparams, intercept, coefs = elastic_net_OLS(train, valid, test)
    print('---------------')
    print(rmse_enet)
    print('---------------')
    print(hyperparams)
    print('---------------')
    print(intercept)
    print('---------------')
    print(coefs)

    model_results.append({
        "model": "Elastic Net OLS",
        "rmse": rmse_enet,
        "details": hyperparams
    })

    # Retrieve results from Frequentist Random Forest
    print("Training freq_random_forest..")

    rmse_rf, hyper, feature_num, feature_names = freq_random_forest(train, valid, test)
    print('---------------')
    print(rmse_rf)
    print('---------------')
    print(hyper)
    print('---------------')
    print(feature_num)
    print('---------------')
    print(feature_names)
    model_results.append({
        "model": "Random Forest (Frequentist)",
        "rmse": rmse_rf,
        "details": hyper
    })

    # # Retrieve results from Bayesian Random Forest
    # print("Training bayesian_random_forest..")
    #
    # rmse_brf, hyper, feat_num, feat_names = bayesian_random_forest(train, valid, test)
    # print('---------------')
    # print(rmse_brf)
    # print('---------------')
    # print(hyper)
    # print('---------------')
    # print(feat_num)
    # print('---------------')
    # print(feat_names)
    #
    # model_results.append({
    #     "model": "Random Forest (Bayesian)",
    #     "rmse": rmse_brf,
    #     "details": hyper
    # })
    #
    # # Retrieve results from Frequentist Extreme Random Forest
    # print("Training freq_ext_rf..")
    #
    # rmse_ext, hyper, feat_num, feat_names = freq_ext_rf(train, valid, test)
    # print('---------------')
    # print(rmse_ext)
    # print('---------------')
    # print(hyper)
    # print('---------------')
    # print(feat_num)
    # print('---------------')
    # print(feat_names)
    #
    # model_results.append({
    #     "model": "Extreme Random Forest (Frequentist)",
    #     "rmse": rmse_ext,
    #     "details": hyper
    # })
    #
    # # Retrieve results from Bayesian Extreme Random Forest
    # print("Training Payes_ext_rf...")
    #
    # rmse_bext, hyper, feat_num, feat_names = bayes_ext_rf(train, valid, test)
    # print('---------------')
    # print(rmse_bext)
    # print('---------------')
    # print(hyper)
    # print('---------------')
    # print(feat_num)
    # print('---------------')
    # print(feat_names)
    #
    # model_results.append({
    #     "model": "Extreme Random Forest (Bayesian)",
    #     "rmse": rmse_bext,
    #     "details": hyper
    # })
    #

    ##new pytorch ANN
    print("Training PyTorch ANN...")

    ##unscaled not use
    # ann_model, ann_val_rmse = train_ann(X_train, y_train, X_valid, y_valid)
    # print("Best Validation RMSE from ANN:", ann_val_rmse)
    #
    # ann_test_rmse, ann_pred = evaluate_ann(ann_model, X_test, y_test)
    # print("Test RMSE from ANN:", ann_test_rmse)
    ##unscaled not use

    ann_model, ann_val_rmse, ann_scaler = train_ann(X_train, y_train, X_valid, y_valid)
    ann_test_rmse, ann_pred = evaluate_ann(ann_model, X_test, y_test, ann_scaler)

    model_results.append({
        "model": "PyTorch ANN",
        "rmse": ann_test_rmse,
        "details": {"val_rmse": ann_val_rmse}
    })

    ###new pytorch BNN
    print("Training PyTorch BNN...")
    test_rmse_bnn, epistemic_unc, aleatoric_unc, bnn_model = BNN(
        (X_train, y_train),
        (X_valid, y_valid),
        (X_test, y_test)
    )

    print("BNN Test RMSE:", test_rmse_bnn)
    print("Epistemic uncertainty (mean):", epistemic_unc.mean())
    print("Aleatoric uncertainty:", aleatoric_unc)

    model_results.append({
        "model": "PyTorch Bayesian Neural Network",
        "rmse": float(test_rmse_bnn),
        "details": {
            "epistemic_mean": float(epistemic_unc.mean()),
            "aleatoric": float(aleatoric_unc)
        }
    })
    # Retrieve results from XGBoost and feature importance plots
    model, rmse_xgb, hyper_xgb, data_gain, data_weight = XGBoost(train, valid, test)
    print('---------------')
    print(rmse_xgb)
    print('---------------')
    print(hyper_xgb)
    print('---------------')

    data_gain.nlargest(40, columns='score').plot(kind='barh', figsize=(20, 10))
    print('---------------')
    data_weight.nlargest(40, columns='score').plot(kind='barh', figsize=(20, 10))

    model_results.append({
        "model": "XGBoost",
        "rmse": rmse_xgb,
        "details": hyper_xgb
    })

    results_df = pd.DataFrame(model_results)
    results_df = results_df.sort_values("rmse", ascending=True).reset_index(drop=True)

    print("\n=======================")
    print("  MODEL PERFORMANCE SUMMARY")
    print("=======================\n")

    print(results_df)

    print(f"🎯 Model selection complete for GEO={geo_filter}")
    return results_df



# Machine Learning and Deep Learning modeling
# Split target variable from feature data matrix
for yr in [2022,2023,2024,2025]:
    listings = pd.read_csv(
        rf"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\P95_feature1_{yr}09.csv",
        index_col=0
    )
    print(listings.columns)
    print(listings.shape)

    ##second options
    import json
    with open(r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\listings_home.geojson", "r") as f:
        geo_ids = json.load(f)



    feature_list = ['none']#'none', 'n_generalParent']#'n_specific', 'n_total'
    poi_options = [False]#, True]
    geojson_options = [False]#, True]

    results = []
    shap_gain_tables = {}
    for feature in feature_list:
        for geo in geojson_options:
            for poi in poi_options:

                df = listings.copy()

                if geo:
                    df = df[df["listing_id"].isin(geo_ids)]

                df_model = prepare_dataset(df, feature,poi)
                print(listings.shape)
                print(listings.columns)

                save_prefix = f"{feature}_geo{geo}_poi{poi}"
                print(f"\n🚀 Running XGBoost Experiment → {save_prefix}")


                run_model_selection = (
                    feature == "none" and
                    poi is False
                )

                if run_model_selection:
                    print("\n🔥 Running FULL MODEL SELECTION for baseline...")

                    prefix = f"{feature}_geo{geo}_poi{poi}"
                    print("Running →", prefix)
                    # use the same df_model
                    result_selection=run_full_model_selection(df_model, geo)
                    result_selection.to_csv(rf'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\reviews_modelSelection\{prefix}_P95_{yr}_noDrop_fixANN.csv')

                    print("🔥 Finished model selection.\n")




