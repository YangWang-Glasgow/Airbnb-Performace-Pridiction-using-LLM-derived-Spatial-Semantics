# Required packages
import io
import shap
import random
import requests
import datetime
import lime
import lime.lime_tabular
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy import stats
from fitter import Fitter
from scipy.stats import pareto
from scipy.stats import median_test
from scipy.spatial.distance import cdist
from numpy.random import default_rng
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os


import statsmodels.api as sm
from scipy.interpolate import interp1d

def read_url(link):
    """ Creates a Pandas Dataframe from online data

    - Parameters:
        - link = Link to the zipped data

    - Output:
        - Pandas Dataframe
    """
    # Define URL and extract information
    # response = requests.get(link)
    # content = response.content

    # Convert into a Pandas Dataframe
    data = pd.read_csv(link, compression='gzip')
    return data


def one_hot_encoder(word, data, text, aux):
    """ Creates binary variables based on the presence of a word in a comment

    - Parameters:
        - word = word to be identified in the text
        - data = dataframe to look in
        - text = column where the text is stored
        - aux = additional word to increase identification accuracy

    - Output:
        - Binary variable
    """
    data[word] = data[text].apply(lambda x: 1 if (word in x.lower() or aux in x.lower()) else 0)

def calculate_days(x, y):
    """
    Vectorized: returns (x - y) in days.
    x and y are Pandas Series of date strings.
    """
    x_dt = pd.to_datetime(x, errors="coerce")
    y_dt = pd.to_datetime(y, errors="coerce")

    # replace missing y with x (0 days difference)
    y_dt = y_dt.fillna(x_dt)

    return (x_dt - y_dt).dt.days


def calculate_days_old(x, y):
    """ Retrieves number of days between given dates

    - Parameters:
        - x = Current date to serve as reference
        - y = Date from which to calculate days

    - Output:
        - Vector of elapsed days between dates
    """
    res = []
    y = y.replace(np.nan, x[0])
    for i in range(len(y)):
        d1 = datetime.datetime.strptime(y[i], '%Y-%m-%d')
        d2 = datetime.datetime.strptime(x[i], '%Y-%m-%d')
        delta = d2 - d1
        res.append(delta.days)
    return res


def new_id(data):
    """ Reset row index for a given dataset

    - Parameters:
        - data = Dataset whose indexes are to be reset

    - Output:
        - Newly indexed dataset
    """
    data = data.reset_index(drop=True)
    return data


def type_sort(data):
    """ Sorts a dataframe by variable type and counts feature classes

    - Parameters:
        -data = Unsorted Pandas Dataframe

    - Output:
        - X = Sorted Pandas Dataframe
        - p1 = Number of quantitative variables
        - p2 = Number of binary variables
        - p3 = Number of multi variables
    """
    # Define data types
    binary = [i for i in data.columns if len(data[i].unique()) == 2]
    multi = [k for k in data.columns if data[k].dtype == 'category']
    quant = [j for j in data.columns if j not in binary and j not in multi]

    # Preparing output
    variables = quant + binary + multi
    X = pd.DataFrame(data.loc[:, variables])
    p1, p2, p3 = len(quant), len(binary), len(multi)
    return X, p1, p2, p3


def vgeom(D):
    """ Computes Geometric Variability

    - Parameters:
        - D = Squared Distance Matrix

    - Output:
        - Geometric Variability
    """
    n = D.shape[0]
    suma = np.sum(D, axis=1)
    return np.sum(suma)/2*n**2


def gower(X1, X2, X3):
    """ Computes statistical distance matrix for mixed data types

    - Parameters:
        - X1 = Continuous features matrix
        - X2 = Binary features matrix
        - X3 = Categorical features matrix

    - Output:
        - Gower Distance Matrix
    """
    # Distance Matrix for continuous variables
    S_inv = np.linalg.inv(np.cov(X1.T, bias=False))
    M = pairwise_distances(X1, metric='mahalanobis', n_jobs=-1, VI=S_inv)
    M2 = np.multiply(M, M)
    M /= vgeom(M2)

    # Distance Matrix for binary variables
    J = cdist(X2, X2, 'jaccard')
    J2 = np.multiply(J, J)
    J /= vgeom(J2)

    # Distance Matrix for categorical variables
    C = cdist(X3, X3, 'hamming')
    C2 = np.multiply(C, C)
    C /= vgeom(C2)

    # Gower Distance Matrix
    D = M + J + C
    return D


def KNN_Imputer_original(feature, target, k):
    """ Performs imputation based on K-NN Algorithm

    - Parameters:
        - feature = Data matrix with features
        - target = Vector containing target variable
        - k = Number of neighbors to consider

    - Output:
        - Non-missing feature data matrix
    """
    # Coerce into numeric data
    feature['target'] = target
    feature = feature.apply(pd.to_numeric, errors='coerce')

    # Dataset with no missing observations
    missing_cols = [i for i in feature.columns if feature[i].isna().sum() > 0]
    distance_data = feature.drop(missing_cols, axis=1)

    # Sort dataset by variable type
    distance_data, p1, p2, p3 = type_sort(distance_data)

    # Create three matrices based on data type
    p = p1 + p2 + p3
    X1 = distance_data.iloc[:, 0:p1] # continuous variables
    X2 = distance_data.iloc[:, p1:p1+p2] # binary variables
    X3 = distance_data.iloc[:, p1+p2:p] # categorical variables

    # Compute Gower Distance Matrix and replace diagonal by maximum value
    D = gower(X1, X2, X3)
    I = np.identity(D.shape[0])
    D = D + I*D.max()

    # Identify k-nearest neighbors based on Gower Distance
    KNN = np.argpartition(D, kth=k, axis=-1)

    # Replace missing values with the mean of neighbors
    for j in missing_cols:
        a = feature[j].isna()
        ids = [i for i in range(len(a)) if a[i] == True]
        for k in ids:
            closest = KNN[k, :5]
            feature.loc[k, j] = np.nanmean(feature.loc[closest, j])

    # In case some neighbors were also missing, perform iterative imputation
    if any(feature.isna().sum() > 0):
        imp = IterativeImputer(estimator=BayesianRidge(),
                           max_iter=25, random_state=42)
        imp.fit(feature)
        feature = imp.transform(feature)

    feature = feature.drop('target', axis=1)
    return feature
def gower_safe(X1, X2, X3):
    """ Safe Gower distance that avoids singular matrices. """

    # ---------------------------
    # Continuous (Mahalanobis)
    # ---------------------------
    if X1.shape[1] > 0:
        # Compute covariance
        cov = np.cov(X1.T, bias=False)

        # FIX: regularize covariance to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-8

        S_inv = np.linalg.inv(cov)

        M = pairwise_distances(X1, metric='mahalanobis', VI=S_inv)
        M2 = M**2
        M /= vgeom(M2)
    else:
        M = 0

    # ---------------------------
    # Binary (Jaccard)
    # ---------------------------
    if X2.shape[1] > 0:
        J = cdist(X2, X2, metric='jaccard')
        J2 = J**2
        J /= vgeom(J2)
    else:
        J = 0

    # ---------------------------
    # Categorical (Hamming)
    # ---------------------------
    if X3.shape[1] > 0:
        C = cdist(X3, X3, metric='hamming')
        C2 = C**2
        C /= vgeom(C2)
    else:
        C = 0

    # Combine distances
    return M + J + C


def KNN_Imputer(feature, target, k):
    """ Robust KNN imputer using Gower distance """

    feature = feature.copy()

    # -----------------------------------------
    # SPLIT NUMERIC AND NON-NUMERIC
    # -----------------------------------------
    num_cols = feature.select_dtypes(include=[np.number]).columns
    non_num_cols = feature.columns.difference(num_cols)

    X = feature[num_cols].copy()

    # Identify columns that contain missing values
    missing_cols = [col for col in X.columns if X[col].isna().sum() > 0]

    # Columns with no missing values
    distance_data = X.drop(columns=missing_cols)

    # Type-sorting for your Gower logic
    distance_data, p1, p2, p3 = type_sort(distance_data)
    p = p1 + p2 + p3

    # Split by type
    X1 = distance_data.iloc[:, :p1]      # continuous
    X2 = distance_data.iloc[:, p1:p1+p2] # binary
    X3 = distance_data.iloc[:, p1+p2:p]  # categorical

    # Remove constant continuous columns
    if p1 > 0:
        non_constant = X1.columns[X1.var() > 0]
        X1 = X1[non_constant]

    # Compute Gower distance safely
    D = gower_safe(X1, X2, X3)
    D = D + np.eye(D.shape[0]) * D.max()

    k_neighbors = np.argpartition(D, kth=k, axis=1)[:, :k]

    def KNN_Imputer(feature, target, k):
        """ Robust KNN imputer using Gower distance """

        feature = feature.copy()

        # -----------------------------------------
        # SPLIT NUMERIC AND NON-NUMERIC
        # -----------------------------------------
        num_cols = feature.select_dtypes(include=[np.number]).columns
        non_num_cols = feature.columns.difference(num_cols)

        X = feature[num_cols].copy()

        # Identify columns that contain missing values
        missing_cols = [col for col in X.columns if X[col].isna().sum() > 0]

        # Columns with no missing values
        distance_data = X.drop(columns=missing_cols)

        # Type-sorting for your Gower logic
        distance_data, p1, p2, p3 = type_sort(distance_data)
        p = p1 + p2 + p3

        # Split by type
        X1 = distance_data.iloc[:, :p1]  # continuous
        X2 = distance_data.iloc[:, p1:p1 + p2]  # binary
        X3 = distance_data.iloc[:, p1 + p2:p]  # categorical

        # Remove constant continuous columns
        if p1 > 0:
            non_constant = X1.columns[X1.var() > 0]
            X1 = X1[non_constant]

        # Compute Gower distance safely
        D = gower_safe(X1, X2, X3)
        D = D + np.eye(D.shape[0]) * D.max()

        k_neighbors = np.argpartition(D, kth=k, axis=1)[:, :k]

        # Impute numeric columns
        for col in missing_cols:
            missing_idx = X[col].isna()
            for row in np.where(missing_idx)[0]:
                neighbors = k_neighbors[row]
                X.loc[row, col] = np.nanmean(X.loc[neighbors, col])

        # Backup iteration if still missing
        if X.isna().any().any():
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=25, random_state=42)
            X[num_cols] = imp.fit_transform(X)

        # -----------------------------------------
        # RECOMBINE WITH NON-NUMERIC COLUMNS
        # -----------------------------------------
        X_final = pd.concat([X, feature[non_num_cols]], axis=1)
        X_final = X_final[feature.columns]

        return X_final

    # Backup iteration if still missing
    if X.isna().any().any():
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=25, random_state=42)
        X[num_cols] = imp.fit_transform(X)

    # -----------------------------------------
    # RECOMBINE WITH NON-NUMERIC COLUMNS
    # -----------------------------------------
    X_final = pd.concat([X, feature[non_num_cols]], axis=1)
    X_final = X_final[feature.columns]

    return X_final

###older version
# def KNN_Imputer(feature, target, k):
#     """ Robust KNN imputer using Gower distance """
#
#     # Coerce into numeric data
#     feature = feature.copy()
#     feature['target'] = target
#     feature = feature.apply(pd.to_numeric, errors='coerce')
#
#     # Columns that contain missing values
#     missing_cols = [col for col in feature.columns if feature[col].isna().sum() > 0]
#
#     # Columns with complete data
#     distance_data = feature.drop(missing_cols, axis=1)
#
#     # Sort dataset by variable type
#     distance_data, p1, p2, p3 = type_sort(distance_data)
#     p = p1 + p2 + p3
#
#     # Split by type
#     X1 = distance_data.iloc[:, :p1]  # continuous
#     X2 = distance_data.iloc[:, p1:p1 + p2]  # binary
#     X3 = distance_data.iloc[:, p1 + p2:p]  # categorical
#
#     # ------------------------------------------------------------------
#     # FIX 1: Remove zero-variance continuous variables (prevents singular matrix)
#     # ------------------------------------------------------------------
#     if p1 > 0:
#         non_constant = X1.columns[X1.var() > 0]
#         X1 = X1[non_constant]
#
#     # ------------------------------------------------------------------
#     # FIX 2: Build robust Gower distance that never errors
#     # ------------------------------------------------------------------
#     D = gower_safe(X1, X2, X3)  # <--- NEW SAFE FUNCTION BELOW
#
#     # Replace diagonal to avoid zero-distance self-neighbors
#     D = D + np.eye(D.shape[0]) * D.max()
#
#     # ------------------------------------------------------------------
#     # Identify k-nearest neighbors
#     # ------------------------------------------------------------------
#     k_neighbors = np.argpartition(D, kth=k, axis=1)[:, :k]
#
#     # ------------------------------------------------------------------
#     # Impute missing values
#     # ------------------------------------------------------------------
#     for col in missing_cols:
#         missing_idx = feature[col].isna()
#         for row in np.where(missing_idx)[0]:
#             neighbors = k_neighbors[row]
#             feature.loc[row, col] = np.nanmean(feature.loc[neighbors, col])
#
#     # ------------------------------------------------------------------
#     # Backup iterative imputation if still missing
#     # ------------------------------------------------------------------
#     if feature.isna().any().any():
#         imp = IterativeImputer(estimator=BayesianRidge(), max_iter=25, random_state=42)
#         feature = pd.DataFrame(imp.fit_transform(feature), columns=feature.columns)
#
#     return feature.drop("target", axis=1)

def imputer_performance(data, imputer, random_cols, random_rows):
    """
    Evaluate an imputation algorithm using a deterministic missing-value mask.

    Parameters:
    - data: [X, y]
    - imputer: 'knn', 'bayes', 'forest'
    - random_cols: list of column names to corrupt
    - random_rows: list/array of row indices to corrupt

    Returns:
    - performance: Series giving reconstruction error per column
    """

    X = data[0].copy()  # features
    y = data[1]         # target

    # Keep a clean copy of original values
    original_data = X.copy()

    # Apply mask: introduce missing values deterministically
    X.loc[random_rows, random_cols] = np.nan

    # Run the selected imputation method
    if imputer == 'knn':
        imputed = KNN_Imputer(X.copy(), y, k=6)

    elif imputer == 'bayes':
        imputed, _ = Iterative_Imputer(X.copy(), y, model='bayesian')

    elif imputer == 'forest':
        imputed, _ = Iterative_Imputer(X.copy(), y, model='forest')

    else:
        raise ValueError("Imputer must be 'knn', 'bayes', or 'forest'.")

    # Compute reconstruction error only on the masked cells
    diff = original_data.loc[random_rows, random_cols] - imputed.loc[random_rows, random_cols]
    performance = diff.abs().sum()

    return performance

def Iterative_Imputer(feature, target, model):
    feature = feature.copy()

    num_cols = feature.select_dtypes(include=[np.number]).columns
    non_num_cols = feature.columns.difference(num_cols)

    # Construct dataset used for imputation
    df = pd.concat([feature[num_cols], target.rename("target")], axis=1)

    # Choose model
    if model == 'bayesian':
        estimator = BayesianRidge()
    else:
        estimator = RandomForestRegressor(
            max_depth=12, bootstrap=True, max_samples=0.5,
            n_jobs=-1, random_state=42
        )

    imp = IterativeImputer(estimator=estimator, max_iter=25, random_state=42)

    df_imputed = pd.DataFrame(
        imp.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    # Remove target
    df_imputed = df_imputed.drop("target", axis=1)

    # Recombine
    final = pd.concat([df_imputed, feature[non_num_cols]], axis=1)
    final = final[feature.columns]

    return final, target

# ###older version
# def Iterative_Imputer(feature, target, model):
#     """ Performs imputation based on Iterative Imputing Algorithm
#
#     - Parameters:
#         - feature = Data matrix with features
#         - target = Vector containing target variable
#         - model = Either Bayesian Ridge Regression or Random Forest
#
#     - Output:
#         - Non-missing feature data matrix
#     """
#     # Store column names
#     cols = feature.columns.tolist()
#     cols.append('target')
#
#     # Coerce into numeric data
#     feature['target'] = target
#     feature = feature.apply(pd.to_numeric, errors='coerce')
#
#     # Select model for Iterative Imputation Algorithm
#     if model == 'bayesian':
#         # Fit a Bayesian Ridge Regression model
#         imp = IterativeImputer(estimator=BayesianRidge(),
#                                max_iter=25, random_state=42)
#         imp.fit(feature)
#         feature = imp.transform(feature)
#
#         # Retrieve a pandas dataframe
#         feature = pd.DataFrame(feature, columns=cols)
#         feature = feature.drop('target', axis=1)
#
#     else:
#         # Fit a Random Forest model
#         random_forest = RandomForestRegressor(max_depth=12, bootstrap=2,
#                                               max_samples=0.5, n_jobs=-1, random_state=42)
#         imp = IterativeImputer(estimator=random_forest, max_iter=25, random_state=42)
#         imp.fit(feature)
#         feature = imp.transform(feature)
#
#         # Retrieve a pandas dataframe
#         feature = pd.DataFrame(feature, columns=cols)
#         feature = feature.drop('target', axis=1)
#     return feature, target

# def imputer_performance(data, imputer):
#     """ Imputes data for algorithmic performance comparison
#
#     - Parameters:
#         - data = Original dataset to perform imputation
#         - imputer = Data imputation algorithm
#
#     - Output:
#         - imputed_data = Final dataset with non-missing values
#     """
#     # Remove columns with missing values
#     # missing_cols = data[0].columns[data.isnull().any()]
#     print(data[0].head())
#     print(data[1].head())
#
#     missing_cols = data[0].columns[data[0].isnull().any()]
#     print(missing_cols)
#     imputer_data = data[0].drop(missing_cols, axis=1)
#
#     # Make a copy for later comparison
#     original_data = imputer_data.copy()
#
#     # Randomly introduce missing values
#     rng = default_rng()
#     random_cols = rng.choice(imputer_data.columns, size=10, replace=False)
#     imputer_data.loc[3000:, random_cols] = np.nan
#
#     # Impute data with different algorithms
#     if imputer == 'knn':
#         imputed_data = KNN_Imputer(imputer_data, data[1], k=6)
#
#     elif imputer == 'bayes':
#         imputed_data = Iterative_Imputer(imputer_data, data[1], model='bayesian')
#         imputed_data = imputed_data[0]
#
#     else:
#         imputed_data = Iterative_Imputer(imputer_data, data[1], model='forest')
#         imputed_data = imputed_data[0]
#
#     performance = original_data.loc[3000:,random_cols].subtract(
#         imputed_data.loc[3000:,random_cols]).sum()
#     return performance


def XAI_SHAP(model, data, graph, obs):
    """ Computes SHAP values and represents XAI graphs

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - graph = Global or local interpretation
        - obs = Index of data instance to explain

    - Output:
        - XAI graphs and SHAP values
    """
    # Print JavaScript visualizations
    shap.initjs()

    # Create object to calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    if graph == 'global':
        # Global Interpretability (feature importance)
        shap.plots.bar(shap_values, max_display=30)

        # Global Interpretability (impact on target variable)
        shap.summary_plot(shap_values, data, max_display=30)

    else:
        # Local Interpretability (coefficients)
        shap.plots.bar(shap_values[obs], max_display=30)
        shap.plots.waterfall(shap_values[obs], max_display=30)

        # Local Interpretability (force plots)
        shap.plots.force(shap_values[obs])
    return shap_values


def XAI_LIME(model, data, obs):
    """ Represents LIME plot

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - observation = Number of listing to explain

    - Output:
        - LIME plot
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(data.values,
                                                       feature_names=data.columns.values.tolist(),
                                                       verbose=True, mode='regression')

    # Explain the selected listing observation
    exp = explainer.explain_instance(data.values[obs], model.predict, num_features=30)

    # Show the predictions
    return exp.show_in_notebook(show_table=True)


def XAI_PDP_ICE(model, data, var1, var2, ice: bool):
    """ Represents PDP and ICE plots

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - var1 = Index of first feature to explain
        - var2 = Index of second feature to explain
        - ICE = Whether to include ICE plots

    - Output:
        - PDP and ICE plots
    """
    # Create vector for feature comparison
    features = [var1, var2, (var1, var2)]

    # Check if user asks only for PDP plots
    if ice:
        PartialDependenceDisplay.from_estimator(model, data,
                                                features, kind=['both', 'both', 'average'])

    else:
        PartialDependenceDisplay.from_estimator(model, data, features)


def bootstrap_lowess_ci(x, y, reps=100, frac=0.3):
    """
    LOWESS smoother + bootstrap confidence intervals.
    Returns:
        main_fit: array of [x, y_smooth]
        ci_low, ci_high: confidence interval curves
    """
    if len(x) < 20:
        return None, None

    # Main LOWESS fit
    main_fit = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)

    # Bootstrap LOWESS fits
    y_boot = []
    n = len(x)
    for _ in range(reps):
        idx = np.random.choice(n, n, replace=True)
        boot = sm.nonparametric.lowess(y[idx], x[idx], frac=frac, return_sorted=True)
        f = interp1d(boot[:, 0], boot[:, 1], bounds_error=False, fill_value="extrapolate")
        y_boot.append(f(main_fit[:, 0]))

    y_boot = np.vstack(y_boot)

    ci_low = np.percentile(y_boot, 5, axis=0)
    ci_high = np.percentile(y_boot, 95, axis=0)

    return main_fit, (main_fit[:, 0], ci_low, ci_high)


def find_roots(xs, ys):
    """Find zero-crossings in a LOWESS curve."""
    roots = []
    for i in range(len(xs) - 1):
        if ys[i] == 0:
            roots.append(xs[i])
        if ys[i] * ys[i + 1] < 0:  # sign change
            # linear interpolation
            r = xs[i] + (xs[i+1] - xs[i]) * (0 - ys[i]) / (ys[i+1] - ys[i])
            roots.append(r)
    return roots


def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in ("_", "-"))



def plot_advanced_shap_interaction_pair(
    primary_feature_name,
    interacting_feature_name,
    shap_interaction_values,
    X_test,
    save_folder
):
    """
    Generates an advanced SHAP interaction plot for one feature pair.

    Inputs:
        primary_feature_name      - e.g., 'n_total'
        interacting_feature_name  - e.g., 'poi_within_800m'
        shap_interaction_values   - output from shap.TreeExplainer(...).shap_interaction_values(X_test)
        X_test                    - test DataFrame
        save_folder               - directory for saving PNG
    """

    os.makedirs(save_folder, exist_ok=True)

    # Index location of features
    f1 = list(X_test.columns).index(primary_feature_name)
    f2 = list(X_test.columns).index(interacting_feature_name)

    shap_slice = shap_interaction_values[:, f1, f2] * 2  # SHAP convention
    x_values = X_test[primary_feature_name]
    interaction_values = X_test[interacting_feature_name]

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx()

    # Scatter points
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
    points = ax2.scatter(
        x_values,
        shap_slice,
        c=interaction_values,
        cmap=cmap,
        alpha=0.9,
        s=25
    )

    # Distribution background
    counts, edges = np.histogram(x_values, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2
    ax1.bar(centers, counts, width=(edges[1] - edges[0]) * 0.7, color="gray", alpha=0.2)
    ax1.set_ylabel(f"Distribution {primary_feature_name}")

    # LOWESS by median split
    med = interaction_values.median()
    groups = {
        "low": {"mask": interaction_values <= med, "color": "blue"},
        "high": {"mask": interaction_values > med, "color": "red"}
    }

    roots = {}

    for name, info in groups.items():
        mask = info["mask"]
        if mask.sum() < 15:
            continue

        # FIX: remove .values → ensures compatibility for Series AND ndarray
        main_fit, ci = bootstrap_lowess_ci(
            np.asarray(x_values[mask]),
            np.asarray(shap_slice[mask])
        )

        if main_fit is not None:
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color=info["color"], lw=2)
            ax2.fill_between(ci[0], ci[1], ci[2], color=info["color"], alpha=0.15)
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])

    # Mark threshold if both curves have similar roots
    if "low" in roots and "high" in roots:
        for r1 in roots["low"]:
            for r2 in roots["high"]:
                if abs(r1 - r2) < (x_values.max() - x_values.min()) * 0.05:
                    mid = (r1 + r2) / 2
                    ax2.axvline(mid, color="purple", linestyle="--")
                    ax2.text(mid, ax2.get_ylim()[1] * 0.9,
                             f"{mid:.2f}", ha="center",
                             color="white", backgroundcolor="purple")

    # Colorbar
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(points, cax=cax).set_label(interacting_feature_name)

    ax1.set_xlabel(primary_feature_name)
    ax2.set_ylabel("")
    ax2.axhline(0, color="black", linestyle="--")
    plt.tight_layout()

    # Save output
    sanitized = f"{sanitize_filename(primary_feature_name)}_vs_{sanitize_filename(interacting_feature_name)}_p95_dropMore.png"
    plt.savefig(os.path.join(save_folder, sanitized), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved advanced interaction plot → {sanitized}")

def plot_clean_shap_interaction_v2(
    primary_feature,
    interacting_feature,
    shap_interaction_values,
    X_test,
    save_folder
):

    import seaborn as sns
    sns.set_style("whitegrid")

    os.makedirs(save_folder, exist_ok=True)

    f1 = list(X_test.columns).index(primary_feature)
    f2 = list(X_test.columns).index(interacting_feature)

    shap_slice = shap_interaction_values[:, f1, f2] * 2
    x = X_test[primary_feature]
    z = X_test[interacting_feature]

    # median split
    median_z = z.median()
    low_mask  = z <= median_z
    high_mask = z >  median_z

    plt.figure(figsize=(9, 6), dpi=150)

    # --- Histogram (normalized + light) ---
    hist_counts, bins, _ = plt.hist(
        x,
        bins=25,
        color="lightgray",
        alpha=0.35,
        density=True
    )

    # --- Scatter points ---
    plt.scatter(x, shap_slice, c=z, cmap="coolwarm", s=30, alpha=0.8)

    # --- LOWESS fits ---
    for mask, color, label in [
        (low_mask,  "blue", "Low POI"),
        (high_mask, "red",  "High POI")
    ]:
        if mask.sum() > 10:
            lowess_fit = sm.nonparametric.lowess(
                shap_slice[mask], x[mask], frac=0.35
            )
            xs, ys = lowess_fit[:, 0], lowess_fit[:, 1]

            # smooth line
            plt.plot(xs, ys, color=color, linewidth=2.5, label=label)

    # --- Threshold Line ---
    # Find intersection between smoothed curves
    # (optional: only if both LOWESS curves exist)
    # ...

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel(primary_feature, fontsize=12)
    plt.ylabel("SHAP Interaction Value", fontsize=12)

    cbar = plt.colorbar()
    cbar.set_label(interacting_feature)

    plt.legend()
    plt.tight_layout()

    output = os.path.join(save_folder, f"clean_{primary_feature}_interaction.png")
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved → {output}")


# =============================================================================
# ===== 6. Plot advanced SHAP interaction graphs =====
# =============================================================================
print(
    "\n--- Starting advanced interaction plot generation task (final revised version) ---")  # Print a message indicating the start of the advanced plotting task


# Define a function to plot advanced interaction graphs between two features
def plot_advanced_interaction_v3(primary_feature_name, interacting_feature_name, x_values, interaction_feature_values,
                              shap_interaction_slice, save_folder):
    """
    Plots and saves an advanced, information-rich feature interaction SHAP plot.
    This function aims to visualize how the SHAP value of a primary feature is influenced by an interacting feature.
    The plot mainly consists of the following parts:

    1.  **Interaction Scatter Plot**: The primary feature's value is on the X-axis, and the SHAP interaction value is on the Y-axis. The color of the scatter points
        is determined by the value of the interacting feature, using the 'seismic' (blue-white-red) colormap to intuitively show how high or low values of the
        interacting feature affect the primary feature's impact.
    2.  **Grouped Fit Curves**: The interacting feature's data is split into "high value" and "low value" groups based on its median. LOWESS smooth fit curves
        (red and blue solid lines) and their confidence intervals are then plotted for each group. This clearly reveals whether the trend of the primary feature's
        effect changes based on the level of the interacting feature.
    3.  **Common Threshold Calibration**: Automatically calculates and finds a "stable" threshold point where both group's fit curves cross y=0. If found, it is
        marked on the plot with a purple dashed line and a label. This threshold may represent a robust point of effect transition that is not influenced by the
        interacting feature.
    4.  **Background Distribution Plot**: A gray bar chart in the background shows the data distribution of the primary feature, providing a data density
        reference for trend analysis.

    Parameters:
    primary_feature_name (str): Name of the primary feature, to be displayed on the X-axis.
    interacting_feature_name (str): Name of the interacting feature, whose values will determine scatter point color and grouping.
    x_values (pd.Series or np.array): Array of the actual values for the primary feature.
    interaction_feature_values (pd.Series or np.array): Array of the actual values for the interacting feature.
    shap_interaction_slice (np.array): Array of the corresponding SHAP interaction values between the primary and interacting features.
    save_folder (str): Folder path where the generated image files will be saved.
    """
    sanitized_primary, sanitized_interacting = sanitize_filename(primary_feature_name), sanitize_filename(
        interacting_feature_name)  # Sanitize the names of the primary and interacting features
    print(
        f"  -> Plotting: '{primary_feature_name}' (Interacting with: '{interacting_feature_name}')")  # Print which pair of features is being plotted
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)  # Create a new figure and subplot
    ax2 = ax1.twinx()  # Create a second y-axis that shares the x-axis

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
    points = ax2.scatter(x_values, shap_interaction_slice, c=interaction_feature_values,
                         cmap=cmap, alpha=1, s=25, zorder=2,
                         label='sample')  # Plot the scatter plot on ax2, with point color determined by the interacting feature's value

    median_val = interaction_feature_values.median()  # Calculate the median of the interacting feature values
    low_mask, high_mask = interaction_feature_values <= median_val, interaction_feature_values > median_val  # Create two boolean masks based on the median for grouping

    # --- Modification 1: Correct the legend label text ---
    groups = {  # Define configuration for the two groups
        'low': {'mask': low_mask, 'color': 'blue', 'offset': 0.9,
                'label': f' {interacting_feature_name} <= {median_val:.2f}'},
        # Configuration for the 'low' value group, including a dynamically generated legend label
        'high': {'mask': high_mask, 'color': 'red', 'offset': 0.8,
                 'label': f' {interacting_feature_name} > {median_val:.2f}'}  # Configuration for the 'high' value group
    }

    counts, bin_edges = np.histogram(x_values, bins=30)  # Calculate the distribution data for the primary feature
    bin_centers, bin_width = (bin_edges[:-1] + bin_edges[1:]) / 2, bin_edges[1] - bin_edges[
        0]  # Calculate the bin centers and width
    ax1.bar(bin_centers, counts, width=bin_width * 0.7, color='gray', alpha=0.2,
            label='Distribution')  # Plot the distribution histogram on ax1
    ax1.set_ylabel('Distribution', fontsize=12)  # Set the y-axis label for ax1
    ax1.set_ylim(0, counts.max() * 1.1)  # Set the y-axis range for ax1

    fits, roots = {}, {}  # Initialize dictionaries to store fit curves and roots
    for name, info in groups.items():  # Iterate through the 'low' and 'high' value groups
        # x_group, shap_group = x_values[info['mask']], shap_interaction_slice[
        #     info['mask']]  # Get the group data using the mask
        x_group = np.asarray(x_values[info['mask']])
        shap_group = np.asarray(shap_interaction_slice[info['mask']])

        if len(x_group) < 10: continue  # If the group has too few samples, skip it

        main_fit, ci_data = bootstrap_lowess_ci(x_group,
                                                shap_group)  # Perform LOWESS smoothing and confidence interval calculation for this group's data
        if main_fit is not None and ci_data is not None:  # If the calculation is successful
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color=f'dark{info["color"]}', lw=2.5,
                     label=info['label'])  # <-- Plot the smooth curve using the corrected label
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color=info['color'],
                             alpha=0.15)  # Fill the confidence interval for this curve
            fits[name] = main_fit  # Store the fitted curve
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])  # Calculate and store the roots of this curve

    if 'low' in roots and 'high' in roots:  # If roots were found for both the 'low' and 'high' groups
        tolerance = (x_values.max() - x_values.min()) * 0.05  # Define a tolerance to determine if two roots are "close"
        for r_low in roots['low']:  # Iterate through the roots of the 'low' group
            for r_high in roots['high']:  # Iterate through the roots of the 'high' group
                if abs(r_low - r_high) < tolerance:  # If the two roots are very close
                    avg_root = (r_low + r_high) / 2  # Calculate their average
                    ax2.axvline(x=avg_root, color='black', linestyle='--',
                                linewidth=1)  # Draw a purple vertical dashed line at this position to indicate a common threshold
                    ax2.text(avg_root, ax2.get_ylim()[1] * 0.9, f' {avg_root:.2f} ', color='white',
                             backgroundcolor='purple', ha='center', va='center', fontsize=9,
                             bbox=dict(facecolor='purple', edgecolor='none', pad=1))  # Add a text label above the line

    # --- Adjust the spacing of the Color Bar ---
    # Manually create a new Axes at a specific position on the Figure, dedicated to the color bar.
    # This method provides the most precise layout control.
    # The coordinates [left, bottom, width, height] are ratios relative to the entire figure (from 0 to 1).
    #   left=0.975: The left edge of the color bar starts at 97.5% from the figure's left side, placing it to the right of the main plot with some gap.
    #   bottom=0.11, height=0.77: Defines the vertical position and length of the color bar, aligning it vertically with the main plot's axes, which solves the issue of the color bar becoming shorter automatically.
    #   width=0.02: Defines the width of the color bar; this value now solely controls the "thickness" of the color bar.
    cbar_ax = fig.add_axes([0.975, 0.11, 0.02, 0.77])
    # Plot the colorbar into the dedicated axes `cbar_ax` we just created.
    #   points: This is the scatter plot object (returned by ax.scatter), and the color bar will be drawn based on its colormap (cmap) and data range.
    #   cax=cbar_ax: This `cax` parameter is key; it tells matplotlib to "draw the color bar in this specified cbar_ax," instead of letting it automatically find a position.
    cbar = fig.colorbar(points, cax=cbar_ax)

    cbar.set_label(f"Value of {interacting_feature_name}", size=12)  # Set the label for the color bar
    ax1.set_xlabel(f'{primary_feature_name}', fontsize=12)  # Set the x-axis label
    ax2.set_ylabel(f'SHAP Interaction Value', fontsize=12)  # Set the y-axis label for ax2
    # fig.suptitle(f"Interaction: {primary_feature_name} vs {interacting_feature_name}", fontsize=14) # Set the main title for the entire figure
    ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=0)  # Draw the y=0 reference line

    y_max_abs = np.abs(shap_interaction_slice).max() * 1.1  # Calculate the y-axis range for ax2
    ax2.set_ylim(-y_max_abs if y_max_abs > 1e-6 else -1,
                 y_max_abs if y_max_abs > 1e-6 else 1)  # Set the y-axis range for ax2
    ax2.legend(loc='best', fontsize=10)  # Display the legend

    ax1.set_zorder(0);  # Set the z-order of ax1 to the bottom layer
    ax2.set_zorder(1);  # Set the z-order of ax2 to the top layer
    ax2.patch.set_alpha(0)  # Set the background of ax2 to transparent

    output_filename = f"interaction_{sanitized_primary}_vs_{sanitized_interacting}_v3.png"  # Construct the output filename
    full_path = os.path.join(save_folder, output_filename)  # Concatenate the full save path
    fig.savefig(full_path, dpi=200, bbox_inches='tight')  # Save the figure
    plt.close(fig)  # Close the figure to release memory

# Define a function to plot advanced interaction graphs between two features
def plot_advanced_interaction_v4(primary_feature_name, interacting_feature_name, x_values, interaction_feature_values,
                              shap_interaction_slice, save_folder):
    """
    Plots and saves an advanced, information-rich feature interaction SHAP plot.
    This function aims to visualize how the SHAP value of a primary feature is influenced by an interacting feature.
    The plot mainly consists of the following parts:

    1.  **Interaction Scatter Plot**: The primary feature's value is on the X-axis, and the SHAP interaction value is on the Y-axis. The color of the scatter points
        is determined by the value of the interacting feature, using the 'seismic' (blue-white-red) colormap to intuitively show how high or low values of the
        interacting feature affect the primary feature's impact.
    2.  **Grouped Fit Curves**: The interacting feature's data is split into "high value" and "low value" groups based on its median. LOWESS smooth fit curves
        (red and blue solid lines) and their confidence intervals are then plotted for each group. This clearly reveals whether the trend of the primary feature's
        effect changes based on the level of the interacting feature.
    3.  **Common Threshold Calibration**: Automatically calculates and finds a "stable" threshold point where both group's fit curves cross y=0. If found, it is
        marked on the plot with a purple dashed line and a label. This threshold may represent a robust point of effect transition that is not influenced by the
        interacting feature.
    4.  **Background Distribution Plot**: A gray bar chart in the background shows the data distribution of the primary feature, providing a data density
        reference for trend analysis.

    Parameters:
    primary_feature_name (str): Name of the primary feature, to be displayed on the X-axis.
    interacting_feature_name (str): Name of the interacting feature, whose values will determine scatter point color and grouping.
    x_values (pd.Series or np.array): Array of the actual values for the primary feature.
    interaction_feature_values (pd.Series or np.array): Array of the actual values for the interacting feature.
    shap_interaction_slice (np.array): Array of the corresponding SHAP interaction values between the primary and interacting features.
    save_folder (str): Folder path where the generated image files will be saved.
    """
    sanitized_primary, sanitized_interacting = sanitize_filename(primary_feature_name), sanitize_filename(
        interacting_feature_name)  # Sanitize the names of the primary and interacting features
    print(
        f"  -> Plotting: '{primary_feature_name}' (Interacting with: '{interacting_feature_name}')")  # Print which pair of features is being plotted
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)  # Create a new figure and subplot
    ax2 = ax1.twinx()  # Create a second y-axis that shares the x-axis

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
    points = ax2.scatter(x_values, shap_interaction_slice, c=interaction_feature_values,
                         cmap=cmap, alpha=1, s=25, zorder=2,
                         label='sample')  # Plot the scatter plot on ax2, with point color determined by the interacting feature's value

    median_val = interaction_feature_values.median()  # Calculate the median of the interacting feature values
    low_mask, high_mask = interaction_feature_values <= median_val, interaction_feature_values > median_val  # Create two boolean masks based on the median for grouping

    # --- Modification 1: Correct the legend label text ---
    groups = {  # Define configuration for the two groups
        'low': {'mask': low_mask, 'color': 'blue', 'offset': 0.9,
                'label': f' {interacting_feature_name} <= {median_val:.2f}'},
        # Configuration for the 'low' value group, including a dynamically generated legend label
        'high': {'mask': high_mask, 'color': 'red', 'offset': 0.8,
                 'label': f' {interacting_feature_name} > {median_val:.2f}'}  # Configuration for the 'high' value group
    }

    counts, bin_edges = np.histogram(x_values, bins=30)  # Calculate the distribution data for the primary feature
    bin_centers, bin_width = (bin_edges[:-1] + bin_edges[1:]) / 2, bin_edges[1] - bin_edges[
        0]  # Calculate the bin centers and width
    ax1.bar(bin_centers, counts, width=bin_width * 0.7, color='gray', alpha=0.2,
            label='Distribution')  # Plot the distribution histogram on ax1
    ax1.set_ylabel('SHAP Interaction Value', fontsize=12)  # Set the y-axis label for ax1
    ax1.set_ylim(0, counts.max() * 1.1)  # Set the y-axis range for ax1

    fits, roots = {}, {}  # Initialize dictionaries to store fit curves and roots
    for name, info in groups.items():  # Iterate through the 'low' and 'high' value groups
        # x_group, shap_group = x_values[info['mask']], shap_interaction_slice[
        #     info['mask']]  # Get the group data using the mask
        x_group = np.asarray(x_values[info['mask']])
        shap_group = np.asarray(shap_interaction_slice[info['mask']])

        if len(x_group) < 10: continue  # If the group has too few samples, skip it

        main_fit, ci_data = bootstrap_lowess_ci(x_group,
                                                shap_group)  # Perform LOWESS smoothing and confidence interval calculation for this group's data
        if main_fit is not None and ci_data is not None:  # If the calculation is successful
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color=f'dark{info["color"]}', lw=2.5,
                     label=info['label'])  # <-- Plot the smooth curve using the corrected label
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color=info['color'],
                             alpha=0.15)  # Fill the confidence interval for this curve
            fits[name] = main_fit  # Store the fitted curve
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])  # Calculate and store the roots of this curve

    if 'low' in roots and 'high' in roots:  # If roots were found for both the 'low' and 'high' groups
        tolerance = (x_values.max() - x_values.min()) * 0.05  # Define a tolerance to determine if two roots are "close"
        for r_low in roots['low']:  # Iterate through the roots of the 'low' group
            for r_high in roots['high']:  # Iterate through the roots of the 'high' group
                if abs(r_low - r_high) < tolerance:  # If the two roots are very close
                    avg_root = (r_low + r_high) / 2  # Calculate their average
                    ax2.axvline(x=avg_root, color='black', linestyle='--',
                                linewidth=1)  # Draw a purple vertical dashed line at this position to indicate a common threshold
                    ax2.text(avg_root, ax2.get_ylim()[1] * 0.9, f' {avg_root:.2f} ', color='white',
                             backgroundcolor='purple', ha='center', va='center', fontsize=9,
                             bbox=dict(facecolor='purple', edgecolor='none', pad=1))  # Add a text label above the line

    # --- Adjust the spacing of the Color Bar ---
    # Manually create a new Axes at a specific position on the Figure, dedicated to the color bar.
    # This method provides the most precise layout control.
    # The coordinates [left, bottom, width, height] are ratios relative to the entire figure (from 0 to 1).
    #   left=0.975: The left edge of the color bar starts at 97.5% from the figure's left side, placing it to the right of the main plot with some gap.
    #   bottom=0.11, height=0.77: Defines the vertical position and length of the color bar, aligning it vertically with the main plot's axes, which solves the issue of the color bar becoming shorter automatically.
    #   width=0.02: Defines the width of the color bar; this value now solely controls the "thickness" of the color bar.
    cbar_ax = fig.add_axes([0.975, 0.11, 0.02, 0.77])
    # Plot the colorbar into the dedicated axes `cbar_ax` we just created.
    #   points: This is the scatter plot object (returned by ax.scatter), and the color bar will be drawn based on its colormap (cmap) and data range.
    #   cax=cbar_ax: This `cax` parameter is key; it tells matplotlib to "draw the color bar in this specified cbar_ax," instead of letting it automatically find a position.
    cbar = fig.colorbar(points, cax=cbar_ax)

    cbar.set_label(f"Value of {interacting_feature_name}", size=12)  # Set the label for the color bar
    ax1.set_xlabel(f'{primary_feature_name}', fontsize=12)  # Set the x-axis label
    ax2.set_ylabel(f'', fontsize=12)  # Set the y-axis label for ax2
    # fig.suptitle(f"Interaction: {primary_feature_name} vs {interacting_feature_name}", fontsize=14) # Set the main title for the entire figure
    ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=0)  # Draw the y=0 reference line

    y_max_abs = np.abs(shap_interaction_slice).max() * 1.1  # Calculate the y-axis range for ax2
    ax2.set_ylim(-y_max_abs if y_max_abs > 1e-6 else -1,
                 y_max_abs if y_max_abs > 1e-6 else 1)  # Set the y-axis range for ax2
    ax2.legend(loc='best', fontsize=10)  # Display the legend

    ax1.set_zorder(0);  # Set the z-order of ax1 to the bottom layer
    ax2.set_zorder(1);  # Set the z-order of ax2 to the top layer
    ax2.patch.set_alpha(0)  # Set the background of ax2 to transparent

    output_filename = f"interaction_{sanitized_primary}_vs_{sanitized_interacting}_v4.png"  # Construct the output filename
    full_path = os.path.join(save_folder, output_filename)  # Concatenate the full save path
    fig.savefig(full_path, dpi=200, bbox_inches='tight')  # Save the figure
    plt.close(fig)  # Close the figure to release memory

def plot_advanced_interaction_v5(
    primary_feature_name,
    interacting_feature_name,
    x_values,
    interaction_feature_values,
    shap_interaction_slice,
    save_folder
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import statsmodels.api as sm
    import os

    os.makedirs(save_folder, exist_ok=True)

    print(f"  -> Plotting: {primary_feature_name} × {interacting_feature_name}")

    # ---------------------------
    # Create figure + axes
    # ---------------------------
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=150)
    ax2 = ax1.twinx()

    # ---------------------------
    # Scatter: SHAP interaction values
    # ---------------------------
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["blue", "#4B0082", "red"]
    )

    points = ax2.scatter(
        x_values,
        shap_interaction_slice,
        c=interaction_feature_values,
        cmap=cmap,
        alpha=0.9,
        s=32
    )

    # ---------------------------
    # Split groups by median
    # ---------------------------
    median_val = interaction_feature_values.median()
    low_mask = interaction_feature_values <= median_val
    high_mask = interaction_feature_values > median_val

    groups = {
        "low": {
            "mask": low_mask,
            "color": "blue",
            "label": f"{interacting_feature_name} lower group",
        },
        "high": {
            "mask": high_mask,
            "color": "red",
            "label": f"{interacting_feature_name} higher group",
        },
    }

    # ---------------------------
    # Background histogram
    # ---------------------------
    counts, bin_edges = np.histogram(x_values, bins=30)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax1.bar(centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.7,
            color="gray", alpha=0.25)
    ax1.set_ylabel("Distribution", fontsize=12)
    ax1.set_ylim(0, counts.max() * 1.15)

    # ---------------------------
    # LOWESS curves for both groups
    # ---------------------------
    fits = {}

    for name, info in groups.items():
        mask = info["mask"]
        if mask.sum() < 10:
            continue

        x_group = np.asarray(x_values[mask])
        y_group = np.asarray(shap_interaction_slice[mask])

        lowess_fit = sm.nonparametric.lowess(
            y_group, x_group, frac=0.35, return_sorted=True
        )

        xs = lowess_fit[:, 0]
        ys = lowess_fit[:, 1]

        # Save for intersection detection
        fits[name] = lowess_fit

        ax2.plot(xs, ys, color=info["color"], lw=2.5, label=info["label"])

    # ---------------------------
    # True LOWESS intersection detection
    # ---------------------------
    if "low" in fits and "high" in fits:

        low_curve = fits["low"]
        high_curve = fits["high"]

        # Determine x-range overlap
        x_min = max(low_curve[:, 0].min(), high_curve[:, 0].min())
        x_max = min(low_curve[:, 0].max(), high_curve[:, 0].max())

        x_common = np.linspace(x_min, x_max, 400)
        low_y = np.interp(x_common, low_curve[:, 0], low_curve[:, 1])
        high_y = np.interp(x_common, high_curve[:, 0], high_curve[:, 1])

        diff = low_y - high_y
        sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]

        if len(sign_change) > 0:
            idx = sign_change[0]
            intersection_x = x_common[idx]

            # Draw vertical green dashed line
            ax2.axvline(intersection_x, color="green", linestyle="--", lw=2)

            ax2.text(
                intersection_x,
                ax2.get_ylim()[1] * 0.8,
                f"{primary_feature_name} = {intersection_x:.2f}",
                color="white",
                backgroundcolor="green",
                ha="center",
                fontsize=9
            )

            print(f"✓ Intersection detected at x = {intersection_x:.3f}")
        else:
            print("No intersection found.")

    # ---------------------------
    # Color bar (moved right)
    # ---------------------------
    cbar_ax = fig.add_axes([0.93, 0.11, 0.02, 0.70])  # moved right 0.975, 0.11, 0.02, 0.77
    cbar = fig.colorbar(points, cax=cbar_ax)
    cbar.set_label(f"{interacting_feature_name} (value)", fontsize=11)

    # ---------------------------
    # Final layout and labels
    # ---------------------------
    ax1.set_xlabel(primary_feature_name, fontsize=12)
    ax2.set_ylabel("", fontsize=12)

    ax2.axhline(0, color="black", linestyle="--", lw=1)
    ax2.legend(loc="best", fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.95, 1])  # leave room for color bar

    out_path = os.path.join(
        save_folder,
        f"interaction_{sanitize_filename(primary_feature_name)}_vs_{sanitize_filename(interacting_feature_name)}_v7.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved → {out_path}")

def plot_advanced_interaction_v6(
    primary_feature_name,
    interacting_feature_name,
    x_values,
    interaction_feature_values,
    shap_interaction_slice,
    save_folder
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import statsmodels.api as sm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import os

    os.makedirs(save_folder, exist_ok=True)

    print(f"  -> Plotting: {primary_feature_name} × {interacting_feature_name}")

    # ---------------------------
    # Create figure with constrained layout
    # ---------------------------
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150, constrained_layout=True)
    ax2 = ax1.twinx()

    # ---------------------------
    # Scatter plot
    # ---------------------------
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["blue", "#4B0082", "red"]
    )

    points = ax2.scatter(
        x_values,
        shap_interaction_slice,
        c=interaction_feature_values,
        cmap=cmap,
        alpha=0.9,
        s=32
    )

    # ---------------------------
    # Split groups by median
    # ---------------------------
    median_val = interaction_feature_values.median()
    low_mask = interaction_feature_values <= median_val
    high_mask = interaction_feature_values > median_val

    groups = {
        "low": {"mask": low_mask, "color": "blue",
                "label": f"{interacting_feature_name} lower group"},
        "high": {"mask": high_mask, "color": "red",
                 "label": f"{interacting_feature_name} higher group"},
    }

    # ---------------------------
    # Background distribution
    # ---------------------------
    counts, bin_edges = np.histogram(x_values, bins=30)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax1.bar(centers, counts,
            width=(bin_edges[1] - bin_edges[0]) * 0.7,
            color="gray", alpha=0.25)

    ax1.set_ylabel("Distribution", fontsize=12)

    # ---------------------------
    # LOWESS curves
    # ---------------------------
    fits = {}

    for name, info in groups.items():
        mask = info["mask"]
        if mask.sum() < 10:
            continue

        x_group = np.asarray(x_values[mask])
        y_group = np.asarray(shap_interaction_slice[mask])

        lowess_fit = sm.nonparametric.lowess(y_group, x_group, frac=0.35)
        xs, ys = lowess_fit[:, 0], lowess_fit[:, 1]

        fits[name] = lowess_fit
        ax2.plot(xs, ys, color=info["color"], lw=2.5, label=info["label"])

    # ---------------------------
    # Intersection detection
    # ---------------------------
    if "low" in fits and "high" in fits:

        low_curve = fits["low"]
        high_curve = fits["high"]

        x_min = max(low_curve[:, 0].min(), high_curve[:, 0].min())
        x_max = min(low_curve[:, 0].max(), high_curve[:, 0].max())

        x_common = np.linspace(x_min, x_max, 400)
        low_y = np.interp(x_common, low_curve[:, 0], low_curve[:, 1])
        high_y = np.interp(x_common, high_curve[:, 0], high_curve[:, 1])

        diff = low_y - high_y
        signs = np.where(np.diff(np.sign(diff)) != 0)[0]

        if len(signs) > 0:
            x_int = x_common[signs[0]]

            ax2.axvline(x_int, linestyle="--", color="green", lw=2)

            ax2.text(
                x_int,
                ax2.get_ylim()[1] * 0.85,
                f"{primary_feature_name} = {x_int:.2f}",
                ha="center",
                fontsize=9,
                color="white",
                bbox=dict(facecolor="green", edgecolor="none", pad=3)
            )

    # ---------------------------
    # Smarter colorbar placement
    # ---------------------------
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.4)

    cbar = fig.colorbar(points, cax=cax)
    cbar.set_label(f"{interacting_feature_name} (value)", fontsize=11)

    # ---------------------------
    # Final labels and legend
    # ---------------------------
    ax1.set_xlabel(primary_feature_name, fontsize=12)
    ax2.set_ylabel("SHAP Interaction Value", fontsize=12)

    ax2.axhline(0, color="black", linestyle="--", lw=1)

    # Keep legend inside plot so nothing overflows
    ax2.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.9
    )

    # ---------------------------
    # Save
    # ---------------------------
    out_path = os.path.join(
        save_folder,
        f"interaction_{sanitize_filename(primary_feature_name)}_vs_{sanitize_filename(interacting_feature_name)}_v6.png"
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved → {out_path}")



def plot_shap_interaction_swapped(
    poi_feature,
    n_total_feature,
    shap_interaction_values,
    X_test,
    save_folder
):
    """
    Interaction plot where:
    - X-axis  = POI density (e.g., poi_within_800m)
    - Y-axis  = SHAP interaction with n_total
    - Vertical line shows POI threshold where LOWESS curves intersect

    Equivalent to mirroring the original plot.
    """

    os.makedirs(save_folder, exist_ok=True)

    # ---------------------------
    # Locate feature indices
    # ---------------------------
    f1 = list(X_test.columns).index(poi_feature)
    f2 = list(X_test.columns).index(n_total_feature)

    # SHAP interaction for the pair
    shap_slice = shap_interaction_values[:, f1, f2] * 2

    # X-axis (POI)
    x_values = X_test[poi_feature].astype(float)

    # Used for color grouping (n_total)
    interaction_values = X_test[n_total_feature].astype(float)

    # ---------------------------
    # Plot base setup
    # ---------------------------
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx()

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["blue", "#4B0082", "red"]
    )

    # Scatter
    points = ax2.scatter(
        x_values,
        shap_slice,
        c=interaction_values,
        cmap=cmap,
        s=25,
        alpha=0.9
    )

    # ---------------------------
    # Histogram of POI distribution
    # ---------------------------
    counts, edges = np.histogram(x_values, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2
    ax1.bar(
        centers, counts,
        width=(edges[1] - edges[0]) * 0.7,
        color="gray", alpha=0.25
    )
    ax1.set_ylabel(f"Distribution of {poi_feature}")

    # ---------------------------
    # LOWESS curves split by median n_total
    # ---------------------------
    med = interaction_values.median()
    groups = {
        "low": {"mask": interaction_values <= med, "color": "blue"},
        "high": {"mask": interaction_values > med, "color": "red"}
    }

    roots = {}

    for name, info in groups.items():
        mask = info["mask"]
        if mask.sum() < 15:
            continue

        # LOWESS + bootstrap CI
        main_fit, ci = bootstrap_lowess_ci(
            np.asarray(x_values[mask]),
            np.asarray(shap_slice[mask])
        )

        if main_fit is not None:
            ax2.plot(main_fit[:, 0], main_fit[:, 1], lw=2, color=info["color"])
            ax2.fill_between(
                ci[0], ci[1], ci[2],
                color=info["color"], alpha=0.15
            )
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])

    # ---------------------------
    # Mark POI threshold where curves intersect
    # ---------------------------
    if "low" in roots and "high" in roots:
        for r1 in roots["low"]:
            for r2 in roots["high"]:
                if abs(r1 - r2) < (x_values.max() - x_values.min()) * 0.05:
                    mid = (r1 + r2) / 2
                    ax2.axvline(mid, color="purple", linestyle="--")

                    # label
                    ax2.text(
                        mid,
                        ax2.get_ylim()[1] * 0.9,
                        f"{mid:.2f}",
                        ha="center",
                        color="white",
                        backgroundcolor="purple"
                    )

    # ---------------------------
    # Formatting
    # ---------------------------
    ax1.set_xlabel(poi_feature)
    ax2.set_ylabel("")  # implicit SHAP
    ax2.axhline(0, color="black", linestyle="--")

    # Colorbar
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(points, cax=cax).set_label(n_total_feature)

    plt.tight_layout()

    # Save
    filename = f"{sanitize_filename(poi_feature)}_vs_{sanitize_filename(n_total_feature)}_P95_SWAPPED_dropMore.png"
    plt.savefig(os.path.join(save_folder, filename), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved swapped interaction plot → {filename}")



from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_fast_shap_interaction(
    primary_feature,
    interacting_feature,
    shap_interaction_values,
    X_test,
    save_folder,
    max_points=2000,     # subsample for speed
    smooth_frac=0.25     # smoother LOWESS curve
):
    """
    Much faster SHAP interaction visualisation.
    Removes bootstrap CI. Uses subsampling + vectorised LOWESS smoothing.
    """

    os.makedirs(save_folder, exist_ok=True)

    # Index features
    f1 = X_test.columns.get_loc(primary_feature)
    f2 = X_test.columns.get_loc(interacting_feature)

    shap_slice = shap_interaction_values[:, f1, f2] * 2
    x = X_test[primary_feature].values
    z = X_test[interacting_feature].values

    # ----------------------------------------
    # 1. Subsample for speed
    # ----------------------------------------
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x = x[idx]
        z = z[idx]
        shap_slice = shap_slice[idx]

    # ----------------------------------------
    # 2. Create figure
    # ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    cmap = mcolors.LinearSegmentedColormap.from_list("fastmap", ["blue", "white", "red"])
    sc = ax.scatter(x, shap_slice, c=z, cmap=cmap, s=20, alpha=0.8)

    # ----------------------------------------
    # 3. LOWESS smoothing (fast)
    # ----------------------------------------
    med = np.median(z)
    low_mask = z <= med
    high_mask = z > med

    if low_mask.sum() > 20:
        low_fit = lowess(shap_slice[low_mask], x[low_mask], frac=smooth_frac)
        ax.plot(low_fit[:, 0], low_fit[:, 1], color="blue", lw=2, label="Low group")

    if high_mask.sum() > 20:
        high_fit = lowess(shap_slice[high_mask], x[high_mask], frac=smooth_frac)
        ax.plot(high_fit[:, 0], high_fit[:, 1], color="red", lw=2, label="High group")

    # ----------------------------------------
    # 4. Basic decoration
    # ----------------------------------------
    plt.axhline(0, lw=1, ls="--", color="black")
    plt.xlabel(primary_feature)
    plt.ylabel("SHAP Interaction Value")
    plt.legend()

    # ----------------------------------------
    # 5. Add color bar
    # ----------------------------------------
    cbar = plt.colorbar(sc)
    cbar.set_label(interacting_feature)

    # ----------------------------------------
    # 6. Save
    # ----------------------------------------
    fname = f"{primary_feature}_vs_{interacting_feature}_FAST_p95.png".replace("/", "_")
    path = os.path.join(save_folder, fname)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"✓ Fast interaction plot saved → {path}")