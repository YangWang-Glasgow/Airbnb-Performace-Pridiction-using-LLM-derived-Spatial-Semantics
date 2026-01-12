# Load required libraries
# import pymc3 as pm
import arviz as az
import xgboost as xgb
from types import LambdaType
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler

#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
import numpy as np


def bayesian_regression(train, valid, test):
    """
    ML-style Bayesian Ridge Regression with Hyperopt tuning.

    Parameters:
        train = [X_train, y_train]
        valid = [X_valid, y_valid]
        test  = [X_test,  y_test]

    Returns:
        rmse_test      = Test RMSE
        best_params    = Best hyperparameters found by Hyperopt
        intercept      = Intercept estimate
        coef_vector    = Regression coefficients
    """

    # -----------------------------------------------------
    # 1) Define Hyperparameter Search Space
    # -----------------------------------------------------
    space = {
        "alpha_1":   hp.uniform("alpha_1",   1e-4, 10),
        "alpha_2":   hp.uniform("alpha_2",   1e-4, 10),
        "lambda_1":  hp.uniform("lambda_1",  1e-4, 10),
        "lambda_2":  hp.uniform("lambda_2",  1e-4, 10)
    }

    # -----------------------------------------------------
    # 2) Objective for Hyperopt (minimize validation RMSE)
    # -----------------------------------------------------
    def objective(params):
        model = BayesianRidge(
            alpha_1=params["alpha_1"],
            alpha_2=params["alpha_2"],
            lambda_1=params["lambda_1"],
            lambda_2=params["lambda_2"]
        )

        model.fit(train[0], train[1])
        preds = model.predict(valid[0])

        rmse = mean_squared_error(valid[1], preds, squared=False)

        return {"loss": rmse, "status": STATUS_OK}

    # -----------------------------------------------------
    # 3) Run Bayesian Optimization (TPE)
    # -----------------------------------------------------
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )

    # -----------------------------------------------------
    # 4) Train Final Model Using Best Params
    # -----------------------------------------------------
    final_model = BayesianRidge(
        alpha_1=best_params["alpha_1"],
        alpha_2=best_params["alpha_2"],
        lambda_1=best_params["lambda_1"],
        lambda_2=best_params["lambda_2"]
    )

    final_model.fit(train[0], train[1])

    # -----------------------------------------------------
    # 5) Evaluate on Test Set
    # -----------------------------------------------------
    test_preds = final_model.predict(test[0])
    rmse_test = mean_squared_error(test[1], test_preds, squared=False)

    # -----------------------------------------------------
    # 6) Outputs
    # -----------------------------------------------------
    intercept = final_model.intercept_
    coef_vector = final_model.coef_

    return rmse_test, best_params, intercept, coef_vector


##old not secure
# def bayesian_regression(train, valid, test):
#     """ Estimates a Linear Regression Model with MAP
#
#     - Parameters:
#         - train = Dataset with training features and target
#         - valid = Dataset with validation features and target
#         - test = Dataset with test features and target
#
#     - Output:
#         - rmse = Root Mean Squared Error on test data
#         - hyperparams = Dictionary containing best performing hyperpriors
#         - intercetp = Estimate for regression intercept
#         - coefs = Coefficient estimates
#     """
#     # Define hyperparameter space with probability distributions
#     space = {
#         'alpha_1': hp.uniform('alpha_1', 0.01, 8),
#         'alpha_2': hp.uniform('alpha_2', 0.01, 8),
#         'lambda_1': hp.uniform('lambda_1', 0.01, 8),
#         'lambda_2': hp.uniform('lambda_2', 0.01, 8)
#     }
#
#     def objective(space):
#         # Build Bayesian Linear Regression Model
#         bayes_reg = BayesianRidge(
#             alpha_1=space['alpha_1'], alpha_2=space['alpha_2'],
#             lambda_1=space['lambda_1'], lambda_2=space['lambda_2'])
#
#         # Fit model to training data
#         bayes_reg.fit(train[0], train[1])
#
#         # Make predictions with validation data for hyperparameter tuning
#         pred = bayes_reg.predict(valid[0])
#         rmse = mean_squared_error(valid[1], pred, squared=False)
#         return {'loss': rmse, 'status': STATUS_OK}
#
#     # Start Bayesian Hyperparameter Optimization with TPE Algorithm
#     trials = Trials()
#     hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest,
#                    max_evals=300, trials=trials)
#
#     # Retrieve best performing hyperparameters
#     space = {
#         'alpha_1': hyperparams['alpha_1'],
#         'alpha_2': hyperparams['alpha_2'],
#         'lambda_1': hyperparams['lambda_1'],
#         'lambda_2': hyperparams['lambda_2']
#     }
#
#     # Build final Bayesian Linear Regression Model
#     bayes_reg = BayesianRidge(
#         alpha_1=space['alpha_1'], alpha_2=space['alpha_2'],
#         lambda_1=space['lambda_1'], lambda_2=space['lambda_2'])
#
#     # Fit model to training data
#     bayes_reg.fit(train[0], train[1])
#
#     # Make predictions on test data for unbiased performance estimate
#     pred = bayes_reg.predict(test[0])
#     rmse = mean_squared_error(test[1], pred, squared=False)
#     intercept = bayes_reg.intercept_
#     coefs = bayes_reg.coef_
#     return rmse, hyperparams, intercept, coefs
#
#
# def posterior_weights(train, train_top):
#     """ Samples posterior distribution of Bayesian Linear Regression
#
#     - Parameters:
#         - train = Complete training dataset
#         - train_top = Training dataset with top 20 relevant features
#
#     - Output:
#         - trace_complete = Posterior distribution of weights (complete)
#         - trace_reduced = Posterior distribution of weights (reduced)
#     """
#     # Obtain posterior distribution for complete model
#     with pm.Model() as regression:
#         # Define prior distributions
#         precision = pm.Gamma('precision', alpha=0.0139, beta=7.9963)
#         intercept = pm.Normal('intercept', mu=0, sd=1/precision)
#         beta = pm.Normal('beta', mu=0, sd=1/precision, shape=train[0].shape[1])
#         epsilon = pm.Gamma('epsilon', alpha=6.0225, beta=1.5251)
#
#         # Linear Regression Weights
#         mu = intercept + pm.math.dot(train[0], beta)
#
#         # Feature variable
#         y_hat = pm.Normal('y_hat', mu=mu, sd=1/epsilon, observed=train[1].values)
#
#         # Sample posterior distribution
#         trace_complete = pm.sample(draws=2000,tune=1000, cores=5)
#
#     # Obtain posterior distribution for reduced model
#     with pm.Model() as regression:
#         # Define prior distributions
#         precision = pm.Gamma('precision', alpha=0.0139, beta=7.9963)
#         intercept = pm.Normal('intercept', mu=0, sd=1/precision)
#         beta = pm.Normal('beta', mu=0, sd=1/precision, shape=train_top[0].shape[1])
#         epsilon = pm.Gamma('epsilon', alpha=6.0225, beta=1.5251)
#
#         # Linear Regression Weights
#         mu = intercept + pm.math.dot(train_top[0], beta)
#
#         # Feature variable
#         y_hat = pm.Normal('y_hat', mu=mu, sd=1/epsilon, observed=train_top[1].values)
#
#         # Sample posterior distribution
#         trace_reduced = pm.sample(draws=2000,tune=1000, cores=5)
#     return trace_complete, trace_reduced
##old not secure


def elastic_net_OLS(train, valid, test):
    """ Estimates a Linear Regression Model with OLS (Elastic Net regularization)

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
        - hyperparams = Dictionary containing best performing hyperparameters
        - intercetp = Estimate for regression intercept
        - coefs = Coefficient estimates
    """
    # Define hyperparameter space with probability distributions
    space = {
        'alpha': hp.uniform('alpha', 0.02, 15),
        'l1_ratio': hp.uniform('l1_ratio', 0.02, 1),
        'max_iter': hp.quniform('max_iter', 1000, 2000, 1),
        'warm_start': hp.choice('warm_start', [True, False])
    }

    def objective(space):
        # Build Linear Regression Model
        freq_reg = ElasticNet(
            alpha=space['alpha'], l1_ratio=space['l1_ratio'],
            max_iter=int(space['max_iter']), warm_start=space['warm_start'])

        # Fit model to training data
        freq_reg.fit(train[0], train[1])

        # Make predictions with validation data for hyperparameter tuning
        pred = freq_reg.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}

    # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    trials = Trials()
    hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest,
                   max_evals=300, trials=trials)

    # Retrieve best performing hyperparameters
    space = {
        'alpha': hyperparams['alpha'],
        'l1_ratio': hyperparams['l1_ratio'],
        'max_iter': hyperparams['max_iter'],
        'warm_start': hyperparams['warm_start']
    }

    # Build final Linear Regression Model
    freq_reg = ElasticNet(
        alpha=space['alpha'], l1_ratio=space['l1_ratio'],
        max_iter=int(space['max_iter']), warm_start=space['warm_start'])

    # Fit model to training data
    freq_reg.fit(train[0], train[1])

    # Make predictions on test data for unbiased performance estimate
    pred = freq_reg.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)
    intercept = freq_reg.intercept_
    coefs = freq_reg.coef_
    return rmse, hyperparams, intercept, coefs


def freq_random_forest(train, valid, test):
    """
    Frequentist Random Forest with proper hyperparameter tuning and cross-validation.

    Parameters:
        train = [X_train, y_train]
        valid = [X_valid, y_valid]
        test  = [X_test,  y_test]

    Returns:
        rmse_test      = RMSE on test data (unbiased)
        best_params    = Hyperparameters selected via Hyperopt
        n_features     = Number of features in model
        feature_names  = Feature name order used
    """

    X_train, y_train = train
    X_valid, y_valid = valid
    X_test,  y_test  = test

    # -------------------------------------------------------------
    # 1. Hyperparameter Search Space (scientifically reasonable)
    # -------------------------------------------------------------
    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 500, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.5, 0.7, 1.0]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 0.05)
    }

    # -------------------------------------------------------------
    # 2. Objective function (5-fold CV RMSE)
    # -------------------------------------------------------------
    def objective(params):

        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            ccp_alpha=params['ccp_alpha'],
            random_state=123,
            n_jobs=-1
        )

        # 5-fold CV on TRAIN + VALID (combined)
        X_cv = pd.concat([X_train, X_valid])
        y_cv = pd.concat([y_train, y_valid])

        cv_rmse = -cross_val_score(
            model, X_cv, y_cv,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1
        ).mean()

        return {'loss': cv_rmse, 'status': STATUS_OK}

    # -------------------------------------------------------------
    # 3. Run Bayesian optimisation (TPE)
    # -------------------------------------------------------------
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=80,   # <— 80–100 is enough and fast
        trials=trials,
        rstate=np.random.default_rng(123)
    )

    # Recover categorical choices
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'min_samples_split': int(best['min_samples_split']),
        'min_samples_leaf': int(best['min_samples_leaf']),
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0][best['max_features']],
        'bootstrap': [True, False][best['bootstrap']],
        'ccp_alpha': best['ccp_alpha']
    }

    # -------------------------------------------------------------
    # 4. Fit final model on TRAIN + VALID
    # -------------------------------------------------------------
    X_final = pd.concat([X_train, X_valid])
    y_final = pd.concat([y_train, y_valid])

    final_rf = RandomForestRegressor(
        **best_params,
        random_state=123,
        n_jobs=-1
    )
    final_rf.fit(X_final, y_final)

    # -------------------------------------------------------------
    # 5. Test Set Evaluation (unbiased)
    # -------------------------------------------------------------
    pred = final_rf.predict(X_test)
    rmse_test = mean_squared_error(y_test, pred, squared=False)

    # -------------------------------------------------------------
    # 6. Output
    # -------------------------------------------------------------
    return rmse_test, best_params, final_rf.n_features_in_, final_rf.feature_names_in_



def freq_random_forest_old(train, valid, test):
    """ Builds Machine Learning model with Random Forest Algorithm

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
        - base_estimator = Child estimator template for fitted sub-estimators
        - feature_number = Number of features seen during fit
        - feature_names = Names of features seen during fit
    """
    # Define hyperparameter space with probability distributions
    space = {
        'n_estimators': hp.quniform("n_estimators", 150, 300, 1),
        'max_depth': hp.quniform('max_depth', 100, 1000, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 4, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1),
        'max_features': hp.uniform('max_features', 0.3, 1),
        'min_child_weight': hp.uniform('min_child_weight', 0.1, 0.6),
        'warm_start': hp.choice('warm_start', [True, False]),
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 2.0)
    }

    def objective(space):
        # Set up Random Forest Algorithm
        freq_rf = RandomForestRegressor(n_estimators=int(space['n_estimators']),
                                       max_depth=int(space['max_depth']), bootstrap=True,
                                       min_samples_split=int(space['min_samples_split']),
                                       min_samples_leaf=int(space['min_samples_leaf']),
                                       max_features=space['max_features'], n_jobs=-1,
                                       warm_start=space['warm_start'], ccp_alpha=space['ccp_alpha'])

        # Fit Algorithm to training data
        freq_rf.fit(train[0], train[1])

        # Make predictions with validation data for hyperparameter tuning
        pred = freq_rf.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}

    # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    trials = Trials()
    hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest,
                       max_evals=300, trials=trials)

    # Retrieve best performing hyperparameters
    space = {
        'n_estimators': hyperparams['n_estimators'],
        'max_depth': hyperparams['max_depth'],
        'min_samples_split': hyperparams['min_samples_split'],
        'min_samples_leaf': hyperparams['min_samples_leaf'],
        'max_features': hyperparams['max_features'],
        'min_child_weight': hyperparams['min_child_weight'],
        'warm_start': hyperparams['warm_start'],
        'ccp_alpha': hyperparams['ccp_alpha']
    }

    # Set up final Random Forest Algorithm
    freq_rf = RandomForestRegressor(n_estimators=int(space['n_estimators']),
                                   max_depth=int(space['max_depth']), bootstrap=True,
                                   min_samples_split=int(space['min_samples_split']),
                                   min_samples_leaf=int(space['min_samples_leaf']),
                                   max_features=space['max_features'], n_jobs=-1,
                                   warm_start=space['warm_start'], ccp_alpha=space['ccp_alpha'])

    # Fit Algorithm to training data
    freq_rf.fit(train[0], train[1])

    # Make predictions on test data for unbiased performance estimate
    pred = freq_rf.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)
    feature_number = freq_rf.n_features_in_
    feature_names = freq_rf.feature_names_in_
    return rmse, hyperparams, feature_number, feature_names


def bayesian_random_forest(train, valid, test):
    """ Builds Machine Learning model with Bayesian Random Forest Algorithm

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
        - base_estimator = Child estimator template for fitted sub-estimators
        - feature_number = Number of features seen during fit
        - feature_names = Names of features seen during fit
    """
    # Define hyperparameter space with probability distributions
    space = {
        'n_estimators': hp.quniform("n_estimators", 150, 300, 1),
        'max_depth': hp.quniform('max_depth', 100, 1000, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 4, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1),
        'max_features': hp.uniform('max_features', 0.3, 1),
        'min_child_weight': hp.uniform('min_child_weight', 0.1, 0.6),
        'warm_start': hp.choice('warm_start', [True, False]),
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 2.0)
    }

    def objective(space):
        # Set up Bayesian Random Forest Algorithm
        bayes_rf = RandomForestRegressor(n_estimators=int(space['n_estimators']),
                                         max_depth=int(space['max_depth']), bootstrap=2,
                                         min_samples_split=int(space['min_samples_split']),
                                         min_samples_leaf=int(space['min_samples_leaf']),
                                         max_features=space['max_features'], n_jobs=-1,
                                         warm_start=space['warm_start'], ccp_alpha=space['ccp_alpha'])

        # Fit Algorithm to training data
        bayes_rf.fit(train[0], train[1])

        # Make predictions with validation data for hyperparameter tuning
        pred = bayes_rf.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}

    # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    trials = Trials()
    hyperparams = fmin(fn=objective, space=space,
            algo=tpe.suggest, max_evals=300, trials=trials)

    # Retrieve best performing hyperparameters
    space = {'n_estimators': hyperparams['n_estimators'],
        'max_depth': hyperparams['max_depth'],
        'min_samples_split': hyperparams['min_samples_split'],
        'min_samples_leaf': hyperparams['min_samples_leaf'],
        'max_features': hyperparams['max_features'],
        'min_child_weight': hyperparams['min_child_weight'],
        'warm_start': hyperparams['warm_start'],
        'ccp_alpha': hyperparams['ccp_alpha']
        }

    # Set up final Random Forest Algorithm
    bayes_rf= RandomForestRegressor(n_estimators = int(space['n_estimators']),
                                    max_depth=int(space['max_depth']), bootstrap=2,
                                    min_samples_split=int(space['min_samples_split']),
                                    min_samples_leaf=int(space['min_samples_leaf']),
                                    max_features=space['max_features'], n_jobs=-1,
                                    warm_start=space['warm_start'], ccp_alpha=space['ccp_alpha'])

    # Fit Algorithm to training data
    bayes_rf.fit(train[0], train[1])

    # Make predictions on test data for unbiased performance estimate
    pred = bayes_rf.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)
    feature_number = bayes_rf.n_features_in_
    feature_names = bayes_rf.feature_names_in_
    return rmse, hyperparams, feature_number, feature_names


def freq_ext_rf(train, valid, test):
    """ Builds Machine Learning model with Extreme Random Forest Algorithm

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
        - base_estimator = Child estimator template for fitted sub-estimators
        - feature_number = Number of features seen during fit
        - feature_names = Names of features seen during fit
    """
    n_feat = train[0].shape[1]
    # Define hyperparameter space with probability distributions
    space = {
        'n_estimators': hp.quniform('n_estimators', 340, 600, 1),
        'max_depth': hp.quniform('max_depth', 8, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1),
        'max_features': hp.quniform('max_features', np.sqrt(n_feat), n_feat-10, 1),
        'max_leaf_nodes': hp.quniform('max_leaf_nodes', 200, 500, 1),
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0001, 0.035),
        'max_samples': hp.uniform('max_samples', 0.1, 1)
    }

    def objective(space):
        # Set up Extreme Random Forest Algorithm
        erf_reg = ExtraTreesRegressor(n_estimators=int(space['n_estimators']),
                                      max_depth=int(space['max_depth']), bootstrap=True,
                                      min_samples_split=int(space['min_samples_split']),
                                      min_samples_leaf=int(space['min_samples_leaf']),
                                      max_features=int(space['max_features']), n_jobs=-1,
                                      max_leaf_nodes=int(space['max_leaf_nodes']),
                                      ccp_alpha=space['ccp_alpha'], max_samples=space['max_samples'])

        # Fit Algorithm to training data
        erf_reg.fit(train[0], train[1])

        # Make predictions with validation data for hyperparameter tuning
        pred = erf_reg.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}

    # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    trials = Trials()
    hyperparams = fmin(fn=objective, space=space,
                       algo=tpe.suggest, max_evals=300, trials=trials)

    # Retrieve best performing hyperparameters
    space = {
        'n_estimators': hyperparams['n_estimators'],
        'max_depth': hyperparams['max_depth'],
        'min_samples_split': hyperparams['min_samples_split'],
        'min_samples_leaf': hyperparams['min_samples_leaf'],
        'max_features': hyperparams['max_features'],
        'max_leaf_nodes': hyperparams['max_leaf_nodes'],
        'ccp_alpha': hyperparams['ccp_alpha'],
        'max_samples': hyperparams['max_samples']
    }

    # Set up final Extreme Random Forest Algorithm
    erf_reg = ExtraTreesRegressor(n_estimators=int(space['n_estimators']),
                                  max_depth=int(space['max_depth']), bootstrap=True,
                                  min_samples_split=int(space['min_samples_split']),
                                  min_samples_leaf=int(space['min_samples_leaf']),
                                  max_features=int(space['max_features']), n_jobs=-1,
                                  max_leaf_nodes=int(space['max_leaf_nodes']),
                                  ccp_alpha=space['ccp_alpha'], max_samples=space['max_samples'])

    # Fit Algorithm to training data
    erf_reg.fit(train[0], train[1])

    # Make predictions on test data for unbiased performance estimate
    pred = erf_reg.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)
    feature_number = erf_reg.n_features_in_
    feature_names = erf_reg.feature_names_in_
    return rmse, hyperparams, feature_number, feature_names


def bayes_ext_rf(train, valid, test):
    """ Builds Machine Learning model with Bayesian Extreme Random Forest Algorithm

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
        - base_estimator = Child estimator template for fitted sub-estimators
        - feature_number = Number of features seen during fit
        - feature_names = Names of features seen during fit
    """
    n_feat = train[0].shape[1]
    # Define hyperparameter space with probability distributions
    space = {
        'n_estimators': hp.quniform('n_estimators', 340, 600, 1),
        'max_depth': hp.quniform('max_depth', 8, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1),
        'max_features': hp.quniform('max_features', np.sqrt(n_feat), n_feat-10, 1),
        'max_leaf_nodes': hp.quniform('max_leaf_nodes', 200, 500, 1),
        'ccp_alpha': hp.uniform('ccp_alpha', 0.0001, 0.035),
        'max_samples': hp.uniform('max_samples', 0.1, 1)
    }

    def objective(space):
        # Set up Bayesian Extreme Random Forest Algorithm
        bayes_erf_reg = ExtraTreesRegressor(n_estimators=int(space['n_estimators']),
                                            max_depth=int(space['max_depth']), bootstrap=2,
                                            min_samples_split=int(space['min_samples_split']),
                                            min_samples_leaf=int(space['min_samples_leaf']),
                                            max_features=int(space['max_features']), n_jobs=-1,
                                            max_leaf_nodes=int(space['max_leaf_nodes']),
                                            ccp_alpha=space['ccp_alpha'], max_samples=space['max_samples'])

        # Fit Algorithm to training data
        bayes_erf_reg.fit(train[0], train[1])

        # Make predictions with validation data for hyperparameter tuning
        pred = bayes_erf_reg.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}

    # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    trials = Trials()
    hyperparams = fmin(fn=objective, space=space,
                       algo=tpe.suggest, max_evals=300, trials=trials)

    # Retrieve best performing hyperparameters
    space = {
        'n_estimators': hyperparams['n_estimators'],
        'max_depth': hyperparams['max_depth'],
        'min_samples_split': hyperparams['min_samples_split'],
        'min_samples_leaf': hyperparams['min_samples_leaf'],
        'max_features': hyperparams['max_features'],
        'max_leaf_nodes': hyperparams['max_leaf_nodes'],
        'ccp_alpha': hyperparams['ccp_alpha'],
        'max_samples': hyperparams['max_samples']
    }

    # Set up final Bayesian Extreme Random Forest Algorithm
    bayes_erf_reg = ExtraTreesRegressor(n_estimators=int(space['n_estimators']),
                                        max_depth=int(space['max_depth']), bootstrap=2,
                                        min_samples_split=int(space['min_samples_split']),
                                        min_samples_leaf=int(space['min_samples_leaf']),
                                        max_features=int(space['max_features']), n_jobs=-1,
                                        max_leaf_nodes=int(space['max_leaf_nodes']),
                                        ccp_alpha=space['ccp_alpha'], max_samples=space['max_samples'])

    # Fit Algorithm to training data
    bayes_erf_reg.fit(train[0], train[1])

    # Make predictions on test data for unbiased performance estimate
    pred = bayes_erf_reg.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)
    feature_number = bayes_erf_reg.n_features_in_
    feature_names = bayes_erf_reg.feature_names_in_
    return rmse, hyperparams, feature_number, feature_names


def XGBoost(train, valid, test):

    # -------------------------------------
    # 1. Hyperparameter search space
    # -------------------------------------
    space = {
        'gamma': hp.uniform('gamma', 0.001, 0.004),
        'reg_alpha': hp.uniform('reg_alpha', 0.5, 2),
        'reg_lambda': hp.uniform('reg_lambda', 0.1, 0.5),
        'max_depth': hp.quniform("max_depth", 10, 20, 1),
        'n_estimators': hp.quniform('n_estimators', 400, 550, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.4),
        'min_child_weight': hp.uniform('min_child_weight', 0.1, 0.4),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'early_stopping_rounds': hp.quniform('early_stopping_rounds', 5, 20, 1)
    }

    # -------------------------------------
    # 2. Objective for Hyperopt
    # -------------------------------------
    def objective(space):

        model = xgb.XGBRegressor(
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            gamma=space['gamma'],
            reg_alpha=space['reg_alpha'],
            reg_lambda=space['reg_lambda'],
            colsample_bytree=space['colsample_bytree'],
            min_child_weight=space['min_child_weight'],
            learning_rate=space['learning_rate'],
            early_stopping_rounds=int(space['early_stopping_rounds']),
            enable_categorical=True,
            tree_method="hist",
            eval_metric="rmse",
            n_jobs=-1,
        )

        evaluation = [(train[0], train[1]), (valid[0], valid[1])]

        model.fit(
            train[0], train[1],
            eval_set=evaluation,
            verbose=False
        )

        pred = model.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)

        return {"loss": rmse, "status": STATUS_OK}

    # -------------------------------------
    # 3. Hyperparameter optimization
    # -------------------------------------
    trials = Trials()
    hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=300,
        trials=trials,
    )

    # -------------------------------------
    # 4. Convert hyperopt outputs to correct types
    # -------------------------------------
    space = {
        'gamma': hyperparams['gamma'],
        'reg_alpha': hyperparams['reg_alpha'],
        'reg_lambda': hyperparams['reg_lambda'],
        'colsample_bytree': hyperparams['colsample_bytree'],
        'min_child_weight': hyperparams['min_child_weight'],
        'learning_rate': hyperparams['learning_rate'],
        'max_depth': int(hyperparams['max_depth']),
        'n_estimators': int(hyperparams['n_estimators']),
        'early_stopping_rounds': int(hyperparams['early_stopping_rounds']),
    }

    # -------------------------------------
    # 5. Train final model
    # -------------------------------------
    model = xgb.XGBRegressor(
        n_estimators=space['n_estimators'],
        max_depth=space['max_depth'],
        gamma=space['gamma'],
        reg_alpha=space['reg_alpha'],
        reg_lambda=space['reg_lambda'],
        colsample_bytree=space['colsample_bytree'],
        min_child_weight=space['min_child_weight'],
        learning_rate=space['learning_rate'],
        early_stopping_rounds=space['early_stopping_rounds'],
        enable_categorical=True,
        tree_method="hist",
        eval_metric="rmse",
        n_jobs=-1,
    )

    evaluation = [(train[0], train[1]), (test[0], test[1])]

    model.fit(
        train[0], train[1],
        eval_set=evaluation,
        verbose=False
    )

    # -------------------------------------
    # 6. Compute test RMSE
    # -------------------------------------
    pred = model.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)

    # -------------------------------------
    # 7. Feature importance extraction
    # -------------------------------------
    gain_score = model.get_booster().get_score(importance_type='gain')
    weight_score = model.get_booster().get_score(importance_type='weight')

    data_gain = pd.DataFrame(
        gain_score.values(), index=gain_score.keys(), columns=['score']
    ).sort_values(by='score', ascending=False)

    data_weight = pd.DataFrame(
        weight_score.values(), index=weight_score.keys(), columns=['score']
    ).sort_values(by='score', ascending=False)

    return model, rmse, space, data_gain, data_weight


def XGBoost_old(train, valid, test):
    """ Builds Machine Learning model with XGBoost Algorithm

    - Parameters:
        - train = Dataset with training features and target
        - valid = Dataset with validation features and target
        - test = Dataset with test features and target

    - Output:
        - rmse = Root Mean Squared Error on test data
        - base_estimator = Child estimator template for fitted sub-estimators
        - feature_number = Number of features seen during fit
        - feature_names = Names of features seen during fit
    """
    # Define hyperparameter space with probability distributions
    space = {
        'gamma': hp.uniform('gamma', 0.001, 0.004),
        'reg_alpha': hp.uniform('reg_alpha', 0.5, 2),
        'reg_lambda': hp.uniform('reg_lambda', 0.1, 0.5),
        'max_depth': hp.quniform("max_depth", 10, 20, 1),
        'n_estimators': hp.quniform('n_estimators', 400, 550, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.4),
        'min_child_weight': hp.uniform('min_child_weight', 0.1, 0.4),
        'seed': 0, 'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'early_stopping_rounds': hp.quniform('early_stopping_rounds', 5, 20, 1)
    }

    def objective(space):
        # Set up XGBoost Algorithm
        xgb_reg = xgb.XGBRegressor(n_estimators=int(space['n_estimators']),
                                   max_depth=int(space['max_depth']), gamma=space['gamma'],
                                   reg_alpha=space['reg_alpha'], min_child_weight=space['min_child_weight'],
                                   colsample_bytree=space['colsample_bytree'], reg_lambda=space['reg_lambda'],
                                   early_stopping_rounds=int(space['early_stopping_rounds']), n_jobs=-1,
                                   learning_rate=space['learning_rate'],
                                    enable_categorical = True,
                                    tree_method = "hist"

                                   )

        # Define evaluation dataset
        evaluation = [(train[0], train[1]), (valid[0], valid[1])]

        # Fit Algorithm to training data
        xgb_reg.fit(train[0], train[1], eval_set=evaluation, verbose=False)

        # Make predictions with validation data for hyperparameter tuning
        pred = xgb_reg.predict(valid[0])
        rmse = mean_squared_error(valid[1], pred, squared=False)
        return {'loss': rmse, 'status': STATUS_OK}

    # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    trials = Trials()
    hyperparams = fmin(fn=objective, space=space,
                       algo=tpe.suggest, max_evals=300, trials=trials)

    # Retrieve best performing hyperparameters
    space = {
        'max_depth': hyperparams['max_depth'],
        'reg_alpha': hyperparams['reg_alpha'],
        'gamma': hyperparams['gamma'], 'seed': 0,
        'reg_lambda': hyperparams['reg_lambda'],
        'n_estimators': hyperparams['n_estimators'],
        'learning_rate': hyperparams['learning_rate'],
        'colsample_bytree': hyperparams['colsample_bytree'],
        'min_child_weight': hyperparams['min_child_weight'],
        'early_stopping_rounds': hyperparams['early_stopping_rounds'],
    }

    # Set up final XGBoost Algorithm
    xgb_reg = xgb.XGBRegressor(n_estimators=int(space['n_estimators']),
                               max_depth=int(space['max_depth']), gamma=space['gamma'],
                               reg_alpha=space['reg_alpha'], min_child_weight=space['min_child_weight'],
                               colsample_bytree=space['colsample_bytree'], reg_lambda=space['reg_lambda'],
                               early_stopping_rounds=int(space['early_stopping_rounds']), n_jobs=-1,
                               learning_rate=space['learning_rate'], tree_method='hist',
                                enable_categorical=True
                               )

    # Define evaluation dataset
    evaluation = [(train[0], train[1]), (test[0], test[1])]

    # Fit Algorithm to training data
    xgb_reg.fit(train[0], train[1], eval_set=evaluation, verbose=False)

    # Make predictions on test data for unbiased performance estimate
    pred = xgb_reg.predict(test[0])
    rmse = mean_squared_error(test[1], pred, squared=False)
    # Retrieve feature importance data to understand split decisions
    gain_score = xgb_reg.get_booster().get_score(importance_type='gain')
    weight_score = xgb_reg.get_booster().get_score(importance_type='weight')
    keys_gain, keys_weight = list(gain_score.keys()), list(weight_score.keys())
    vals_gain, vals_weight = list(gain_score.values()), list(weight_score.values())
    data_gain = pd.DataFrame(data=vals_gain, index=keys_gain,
                             columns=['score']).sort_values(by='score', ascending=False)
    data_weight = pd.DataFrame(data=vals_weight, index=keys_weight,
                               columns=['score']).sort_values(by='score', ascending=False)
    return xgb_reg, rmse, hyperparams, data_gain, data_weight
