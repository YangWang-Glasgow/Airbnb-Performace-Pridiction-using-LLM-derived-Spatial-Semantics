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

# # Retrieving data using our defined function
#
# listings = pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\POI\2023_09_ListingPOIwithin800m.csv')
# print(listings.shape)
# print(listings.columns)
#
# # Removing inactive users
# listings = listings.drop(listings[listings['host_since'].isnull()].index)
#
# listings = listings.drop(
#     listings[listings['maximum_nights_avg_ntm'].isnull()].index)
#
# listings = listings.reset_index(drop=True)
#
# # Feature Engineering and Data Cleaning
# # Removing dollar symbol from price and taking logarithms
# price = [y[1:] for y in listings['price']]
# price = [int(float(y.replace(',', ''))) for y in price]
# listings['price'] = [np.log(y) if y != 0 else 0 for y in price]
#
# # Encoding neighborhood clusters
# groups = 'neighbourhood_group_cleansed'
# listings['neighborhood'] = listings[groups].astype('category').cat.codes
#
# # Encoding specific neighborhoods
# groups = 'neighbourhood_cleansed'
# listings['neighborhood_group'] = listings[groups].astype('category').cat.codes
#
# # Imputing empty reviews and creating new feature identifying non-reviewed listings
# listings['inactive'] = listings['reviews_per_month'].replace(np.nan, 'inact')
# listings['inactive'] = [0 if i != 'inact' else 1 for i in listings['inactive']]
# listings['reviews_month'] = listings['reviews_per_month'].replace(np.nan, 0)
# listings = listings.drop('reviews_per_month', axis=1)
#
# # Encoding room type
# listings['room_type'] = listings['room_type'].replace(
#     ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'],
#     [1, 2, 3, 4])
#
# # Encoding host response time and creating new feature identifying non-respondent
# listings['host_response_time'] = listings['host_response_time'].replace(
#     [np.nan, 'within an hour', 'within a few hours', 'within a day', 'a few days or more'],
#     [0, 1, 2, 3, 4])
#
# # Detecting hosts that do not respond
# listings['responds'] = listings['host_response_rate'].replace(
#     [np.nan, '0%'], [0, 0])
#
# listings['responds'] = [1 if i != 0 else 0 for i in listings['responds']]
#
# # Removing '%' symbol from response rate and coercing it into an integer
# for i in ['host_response_rate', 'host_acceptance_rate']:
#     rate = listings[i].replace(np.nan, 0)
#     rate = [x if x == 0 else x.replace('%', '') for x in rate]
#     rate = [int(j) for j in rate]
#     listings[i] = rate
#
# # One-hot encoding several features
# features = [
#     'has_availability', 'instant_bookable', 'host_has_profile_pic',
#     'host_is_superhost', 'host_identity_verified'
# ]
#
# for i in features:
#     listings[i] = listings[i].apply(lambda x: 1 if x == 't' else 0)
#
# # Calculating days since different events
# for j in ['last_review', 'first_review', 'host_since']:
#     listings[j] = calculate_days(listings['last_scraped'], listings[j])
#
# # Identifying number of bathrooms
# print(listings['bathrooms_text'].unique().tolist())
# listings['bathrooms_text'] = listings['bathrooms_text'].replace(
#     ['Private half-bath', np.nan, 'Half-bath', 'Shared half-bath'],
#     ['1 private', '0 baths', '1 private', '1 private'])
#
# listings['bathrooms'] = [int(j[0]) for j in listings['bathrooms_text']]
# listings = listings.drop('bathrooms_text', axis=1)
#
# # Normalized space for longitude and latitude
# lat, lon = listings['latitude'], listings['longitude']
# listings['geo_x'] = np.multiply(np.cos(lat), np.cos(lon))
# listings['geo_y'] = np.multiply(np.cos(lat), np.sin(lon))
# listings['geo_z'] = np.sin(lat)
# listings = listings.drop(['latitude', 'longitude'], axis=1)
#
# # Check whether host supplied information about listing
# listings['description'] = listings['description'].replace(np.nan, 0)
# listings['description'] = [1 if i != 0 else 0 for i in listings['description']]
#
# # Encoding
# listings['property'] = listings['property_type'].astype('category').cat.codes
#
# # Selecting some amenities
# print(listings['amenities'].unique().tolist())
# one_hot_encoder('tv', listings, 'amenities', 'television')
# one_hot_encoder('netflix', listings, 'amenities', 'amazon')
# one_hot_encoder('gym', listings, 'amenities', 'gym')
# one_hot_encoder('elevator', listings, 'amenities', 'lift')
# one_hot_encoder('fridge', listings, 'amenities', 'refrigerator')
# one_hot_encoder('heating', listings, 'amenities', 'heating')
# one_hot_encoder('hair_dryer', listings, 'amenities', 'hair dryer')
# one_hot_encoder('air_conditioning', listings, 'amenities', 'air conditioning')
# one_hot_encoder('hot_tub', listings, 'amenities', 'hot tub')
# one_hot_encoder('oven', listings, 'amenities', 'oven')
# one_hot_encoder('bbq', listings, 'amenities', 'barbecue')
# one_hot_encoder('security cameras', listings, 'amenities', 'camera')
# one_hot_encoder('workspace', listings, 'amenities', 'workspace')
# one_hot_encoder('coffee', listings, 'amenities', 'coffee maker')
# one_hot_encoder('backyard', listings, 'amenities', 'backyard')
# one_hot_encoder('outdoor_dining', listings, 'amenities', 'outdoor dining')
# one_hot_encoder('greets', listings, 'amenities', 'host greets')
# one_hot_encoder('pool', listings, 'amenities', 'pool')
# one_hot_encoder('beachfront', listings, 'amenities', 'beach view')
# one_hot_encoder('patio', listings, 'amenities', 'balcony')
# one_hot_encoder('luggage', listings, 'amenities', 'luggage dropoff')
# one_hot_encoder('furniture', listings, 'amenities', 'outdoor furniture')
#
#
# # Get hosts' gender
# #amended
# d = gender.Detector()
#
# # Apply name gender detector
# listings['name_gender'] = listings['host_name'].apply(lambda x: d.get_gender(x) if isinstance(x, str) else 'unknown')
#
# # Map gender-guesser results to numeric codes
# gender_map = {
#     'female': 0, 'mostly_female': 0,
#     'male': 1, 'mostly_male': 1,
#     'unknown': np.nan, 'andy': np.nan,
#     None: np.nan
# }
# listings['name_gender_code'] = listings['name_gender'].map(gender_map)
#
#
# # ==========================================================
# # 2. Run CV model ONLY for hosts still missing gender
# # ==========================================================
# missing_ids = listings[listings['name_gender_code'].isna()]['id']
# print(len(missing_ids))
#
# cv_data = cv_model(listings[listings['id'].isin(missing_ids)])
# print(listings[listings['id'].isin(missing_ids)].head())
# print(listings[listings['id'].isin(missing_ids)].shape)
#
# # Keep only valid CV predictions
# cv_data['cv_gender'] = cv_data['cv_gender'].apply(lambda x: np.nan if x not in ['Woman', 'Man'] else x)
# cv_data['cv_gender'] = cv_data['cv_gender'].replace(['Woman', 'Man'], [0, 1])
# print(cv_data.head())
# print(cv_data.shape)
#
# # cv_data.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\cvout.csv')
# cv_data=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\cvout.csv')
# print(cv_data.shape)
# #
# # # Merge CV predictions BACK into listings (ONLY missing ones)
# listings = listings.merge(cv_data[['id', 'cv_gender']], on='id', how='left')
#
# # Final gender: use name_gender unless missing → falls back to cv_gender
# listings['gender_final'] = listings['name_gender_code']
# listings.loc[listings['gender_final'].isna(), 'gender_final'] = listings['cv_gender']
#
# # ==========================================================
# # 3. Manual replacements (your list of female names)
# # ==========================================================
# print(listings[listings['gender_final'].isnull()]['host_name'].unique().tolist())
#
# female = [
#     'Abrianna', 'Brinda', 'Denisse', 'Tytyana', 'Susian', 'Diba',
#     'Feven', 'Flor Arely', 'Dipti', 'Shaquetta', 'Jonitha'
# ]
# fix_ids = listings[listings['host_name'].isin(female)].index
# listings.loc[fix_ids, 'gender_final'] = 0  # female = 0
#
# # ==========================================================
# # 4. Impute remaining NaNs using mode
# # ==========================================================
# mode_gender = stats.mode(listings['gender_final'], nan_policy='omit')[0][0]
# listings['gender_final'] = listings['gender_final'].fillna(mode_gender)
#
# # ==========================================================
# # 5. Create categorical version
# # ==========================================================
# print(listings[listings['gender_final'].isnull()]['host_name'].unique().tolist())
#
# listings['gender_final'] = listings['gender_final'].astype('category')
#
# listings.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_lst_clean_2023_09.csv')


# ############older version
# # Get hosts' gender
# # Predict gender with CV Model
# cv_data = cv_model(listings)
#
# # Encode gender predictions
# cv_data['cv_gender'] = cv_data['cv_gender'].apply(
#     lambda x: np.nan if x not in ['Woman', 'Man'] else x)
#
# cv_data['cv_gender'] = cv_data['cv_gender'].replace(['Woman', 'Man'], [0, 1])
#
# # Left join with original Airbnb listings dataset
# listings = listings.merge(
#     cv_data[['id', 'cv_gender']], on='id', how='left')
#
# # NLP Model predictions for hosts with no profile picture
# d = gender.Detector()
# ids = listings[listings['host_has_profile_pic'] == 0].index.tolist()
# listings.loc[ids, 'cv_gender'] = [d.get_gender(listings['host_name'][i]) for i in ids]
#
# listings['cv_gender'] = listings['cv_gender'].replace(
#     ['female', 'male'], [0, 1])
#
# listings['cv_gender'] = listings['cv_gender'].replace(
#     ['mostly_female', 'mostly_male'], [0, 1])
#
# # Define NLP column for later use
# listings['nlp_gender'] = listings['cv_gender']
#
# # Multiple hosts encoding
# listings['cv_gender'] = listings['cv_gender'].replace(np.nan, 2)
#
# # Manually replacing some hosts' gender
# idx = []
# female = [
#     'Abrianna', 'Brinda', 'Denisse', 'Tytyana', 'Susian', 'Diba',
#     'Feven', 'Flor Arely', 'Dipti', 'Shaquetta', 'Jonitha','Edele',
# ]
#
# for i in female:
#     idx.extend(listings['cv_gender'][listings['host_name'] == i].index.tolist())
#
# listings.loc[idx, 'cv_gender'] = 1
#
# # Imputing the rest of hosts with mode gender
# listings['cv_gender'] = listings['cv_gender'].replace(['unknown', 'andy'], [5, 5])
# mode = stats.mode(listings['cv_gender'])[0][0]
# listings['cv_gender'] = listings['cv_gender'].replace(5, mode)
# listings['cv_gender'] = listings['cv_gender'].astype('category')
#
# # Define NLP column for later use
# names = listings['host_name']
# ids = listings[listings['cv_gender'] != 2].index.tolist()
# listings.loc[ids, 'nlp_gender'] = [d.get_gender(names[i]) for i in ids]
#
# listings['nlp_gender'] = listings['nlp_gender'].replace(['female', 'male'], [0, 1])
# listings['nlp_gender'] = listings['nlp_gender'].replace(['mostly_female', 'mostly_male'], [0, 1])
#
# listings.loc[idx, 'nlp_gender'] = 1
#
# listings['nlp_gender'] = listings['nlp_gender'].replace(np.nan, 2)
#
# listings['nlp_gender'] = listings['nlp_gender'].replace(['unknown', 'andy'], [5, 5])
# mode = stats.mode(listings['nlp_gender'])[0][0]
# listings['nlp_gender'] = listings['nlp_gender'].replace(['unknown', 'andy'], [mode, mode])
# listings['nlp_gender'] = listings['nlp_gender'].astype('category')
#
# ############older version



##clean reviews
# Reading reviews and ditching out non-latin symbols
# os.chdir(
#     r'C:\Users\yw30f\OneDrive - University of Glasgow\Airbnb\regulations\InsideAirbnb\data\insideAirbnbScraping_01112024\Edinburgh\hidden')
# reviews = read_url('2023-09-11_reviews.csv.gz')
# print(reviews.columns)
# print(reviews.shape)
# reviews['comments'] = only_latin(reviews['comments'])
# langs = [classify(i)[0] for i in reviews['comments']]
# ids = [j for j in range(len(langs)) if langs[j] != 'en']
# reviews = reviews.drop(labels=ids, axis=0)
# reviews = reviews.reset_index(drop=True)
#
# # Predict sentiment
# reviews['sentiment'] = [sentiment_vader(clean_text(i)) for i in reviews['comments']]
# reviews.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_reviews_2023_09.csv')

###merge step
# # # # ##preprocessed read
# reviews= pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_reviews_2023_09.csv')
# listings= pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_lst_clean_2023_09.csv')
#
#
#
# # # Average sentiment for each listing
# sent_avg = reviews.groupby('listing_id', as_index=False)['sentiment'].mean()
# sent_median = reviews.groupby('listing_id', as_index=False)['sentiment'].median()
# sent_mode = reviews.groupby('listing_id', as_index=False)['sentiment'].agg(
#     lambda x: x.value_counts().index[0])
#
# # Set up columns for later SQL join
# sent_avg.columns = ['id', 'sent_mean']
# sent_median.columns = ['id', 'sent_median']
# sent_mode.columns = ['id', 'sent_mode']
#
# # Add average, median and mode sentiment to original dataset
# listings = listings.merge(sent_median, on='id', how='left')
# print(listings.shape)
# print(listings[listings['sent_median'].isnull()].shape)
# listings = listings.merge(sent_avg, on='id', how='left')
# listings = listings.merge(sent_mode, on='id', how='left')
#
# # Encode mode
# listings['sent_mode'] = listings['sent_mode'].astype('category')
#
# # Identify non-commented listings
# listings['no_review'] = listings['sent_mean'].fillna('nan')
# listings['no_review'] = [0 if i == 'nan' else 1 for i in listings['no_review']]
#
# # Drop unused features
# unused = [
#     'id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'picture_url',
#     'neighborhood_overview', 'host_id', 'host_url', 'host_name', 'host_location',
#     'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
#     'host_total_listings_count', 'host_verifications', 'neighbourhood', 'amenities',
#     'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
#     'maximum_maximum_nights', 'calendar_updated', 'availability_60', 'license',
#     'calendar_last_scraped', 'property_type', 'neighbourhood_group_cleansed',
#     'neighbourhood_cleansed'
# ]
#
# listings = listings.drop(unused, axis=1)
#
# # Correctly encoding categorical features
# categorical = [
#     'property', 'room_type', 'neighborhood', 'neighborhood_group',
#     'host_response_time', 'sent_mode', 'cv_gender', 'gender_final'
# ]
#
# for i in categorical:
#     listings[i] = listings[i].astype('category')
#
# # Remove gender from CV model
# removeGenderCols=[c for c in listings.columns if '_gender' in c ]
# listings = listings.drop(removeGenderCols, axis=1)
# print(listings.columns)
# listings.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2023_09.csv')

# ##we don't need baysian model
# # Hypothesis testing for median price difference between genders
# # Fit distributions to prices set by women
# female_price = listings['price'][listings['nlp_gender'] == 0].values
# f = Fitter(female_price,
#            distributions=['gamma', 'lognorm', 'beta',
#                           'burr', 'norm'])
# f.fit()
# print(f.summary())
#
# # Get parameters from best fit
# params = f.get_best(method='sumsquare_error')
# params = f.get_best(method='aic')
# params = f.get_best(method='bic')
#
# # Scale female prices according to distribution parameters
# scaled_price_fem = (female_price - params['burr']['loc']) / params['burr']['scale']
#
# # Get posterior value for alpha
# alpha_post_fem = 0.001 + len(female_price)
#
# # Get posterior value for beta
# beta_post_fem = 0.001 + \
#                 np.sum(np.log(1 + scaled_price_fem ** -params['burr']['c']))
#
# # Fit distributions to prices set by men
# male_price = listings['price'][listings['nlp_gender'] == 1].values
# f = Fitter(male_price,
#            distributions=['gamma', 'lognorm', 'beta',
#                           'burr', 'norm'])
# f.fit()
# print(f.summary())
#
# # Get parameters from best fit
# params = f.get_best(method='sumsquare_error')
# params = f.get_best(method='aic')
# params = f.get_best(method='bic')
#
# # Scale price according to distribution parameters
# scaled_price_male = (male_price - params['burr']['loc']) / params['burr']['scale']
#
# # Get posterior value for alpha
# alpha_post_male = 0.001 + len(male_price)
#
# # Get posterior value for beta
# beta_post_male = 0.001 + \
#                  np.sum(np.log(1 + scaled_price_male ** -params['burr']['c']))
#
# # Sampling from posterior distributions
# results = []
# for i in range(300):
#     female = pareto.rvs(alpha_post_fem, loc=0, scale=beta_post_fem,
#                         size=1000, random_state=i) / beta_post_fem
#
#     male = pareto.rvs(alpha_post_male, loc=0, scale=beta_post_male,
#                       size=1000, random_state=i) / beta_post_male
#
#     medians = np.median(np.exp(female)) - np.median(np.exp(male))
#     results.append(medians)
#
# # Check if females charge less than males (median value)
# bias = [1 if abs(i) >= 0.01 else 0 for i in results]
# res = np.sum(bias) / len(bias)
# print(res)
#
# # Frequentist sign test
# stat, pval, med, tbl = median_test(female_price, male_price)
# print(pval)
# ##we don't need baysian model



# Machine Learning and Deep Learning modeling
# Split target variable from feature data matrix

listings= pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2023_09.csv',index_col=[0])
print(listings.columns)
print(listings.shape)

# ##second options
# import json
# with open(r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\listings_home.geojson", "r") as f:
#     lst_type = json.load(f)
# listings=listings[listings['listing_id'].isin(lst_type)]
# print(listings.shape)

##read LLM
df_llm=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\listingout_merge_locationcount_onlyIdCount_tidy.csv')
print(df_llm.shape)
print(df_llm.columns)
df_llm=df_llm[['id','n_specific']] #'n_specific', 'n_general', 'n_parent', 'n_total', 'n_generalParent'
listings=listings.merge(df_llm,right_on='id',left_on='listing_id',how='left')
print(listings.shape)
print(listings[listings['n_specific'].isnull()].shape)

##further clean columns
listings=listings.drop(['neighborhood', 'review_scores_accuracy','sent_median', 'sent_mode','Unnamed: 0','Unnamed: 0.1','source','geometry','buffer_800m','listing_id'],axis=1)
##wihtout poi_within_800
# listings=listings.drop(['poi_within_800m'],axis=1)




# listings=listings.fillna(0)

y = listings['price']
X = listings.loc[:, listings.columns != 'price']

# Data partitioning into training, validation and test for ML Modeling
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.09, random_state=123)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, test_size=0.15, random_state=123)

# Reset indexes of each dataset
X_train, X_valid, X_test = new_id(X_train), new_id(X_valid), new_id(X_test)
y_train, y_valid, y_test = new_id(y_train), new_id(y_valid), new_id(y_test)

# Stack features and target
train = [X_train, y_train]
valid = [X_valid, y_valid]
test = [X_test, y_test]

# Check best data imputation algorithm
#checked random forest is the best
###added
rng = default_rng(seed=42)  # fixed seed for reproducibility

# Choose 10 random columns to corrupt
random_cols = rng.choice(X_test.columns, size=10, replace=False)

# Select random rows (10% of test set)
num_rows = int(0.1 * len(X_test))
random_rows = rng.choice(X_test.index, size=num_rows, replace=False)

print("Corrupting columns:", random_cols)
print("Corrupting rows:", random_rows[:10])



knn_imputer = imputer_performance(test, 'knn', random_cols, random_rows)
print(knn_imputer)
bayes_imputer = imputer_performance(test, 'bayes', random_cols, random_rows)
print(bayes_imputer)
forest_imputer = imputer_performance(test, 'forest', random_cols, random_rows)
print(forest_imputer)

print((knn_imputer < bayes_imputer).sum(), "columns where KNN outperforms Bayes")
print((knn_imputer < forest_imputer).sum(), "columns where KNN outperforms Forest")
print((forest_imputer < bayes_imputer).sum(), "columns where Forest outperforms Bayes")


if (forest_imputer < knn_imputer).sum() > 0 and (forest_imputer < bayes_imputer).sum() > 0:
    best_model = "forest"
elif (knn_imputer < bayes_imputer).sum() > (forest_imputer < bayes_imputer).sum():
    best_model = "knn"
else:
    best_model = "bayes"

print("Best imputer:", best_model)

if best_model == "forest":
    X_train, _ = Iterative_Imputer(X_train, y_train, model='forest')
    X_valid, _ = Iterative_Imputer(X_valid, y_valid, model='forest')
    X_test, _ = Iterative_Imputer(X_test, y_test, model='forest')

elif best_model == "knn":
    X_train = KNN_Imputer(X_train, y_train, k=6)
    X_valid = KNN_Imputer(X_valid, y_valid, k=6)
    X_test = KNN_Imputer(X_test, y_test, k=6)

elif best_model == "bayes":
    X_train, _ = Iterative_Imputer(X_train, y_train, model='bayesian')
    X_valid, _ = Iterative_Imputer(X_valid, y_valid, model='bayesian')
    X_test, _ = Iterative_Imputer(X_test, y_test, model='bayesian')



# Impute data with best perfoming algorithm --older
# X_train = forest_imputer(X_train, y_train, k=6)
# X_valid = forest_imputer(X_valid, y_valid, k=6)
# X_test = forest_imputer(X_test, y_test, k=6)
# Impute data with best perfoming algorithm --older


X_train, y_train = Iterative_Imputer(X_train, y_train, model='forest')
X_valid, y_valid =  Iterative_Imputer(X_valid, y_valid, model='forest')
X_test, y_test =  Iterative_Imputer(X_test, y_test, model='forest')

print(listings.columns)

listings.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_afterImputation.csv')

categorical = [
    'property', 'room_type',  'neighborhood_group',
    'host_response_time', 'gender_final'
]
# Correctly encode categorical features
for i in categorical:
    X_train[i] = X_train[i].astype('category')
    X_valid[i] = X_valid[i].astype('category')
    X_test[i] = X_test[i].astype('category')

# Stack final features and target
train = [X_train, y_train]
valid = [X_valid, y_valid]
test = [X_test, y_test]

print ('trainig start --------------')
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
    "details": hyperparams
})

# # Define top features
# top_features = ['bathrooms', 'beds', 'bedrooms', 'property',
#                 'accommodates', 'bbq', 'hot_tub', 'beachfront'
#                  'calculated_host_listings_count_shared_rooms',
#                 'calculated_host_listings_count_private_rooms',
#                 'host_since', 'geo_x', 'geo_y', 'geo_z',
#                 'availability_365', 'neighborhood_group',
#                 'first_review', 'last_review', 'host_listings_count',
#                 'reviews_month'
#                 ]
#
# # Create dataset for reduced model sampling
# train_top = X_train.loc[:, top_features]
#
# # Retrieve posterior distribution of weights
# complete_trace, reduced_trace = posterior_weights(train, train_top)
#
# # Get credibe intervals
# coefs_complete = az.summary(complete_trace, kind='stats')
# print(coefs_complete)
#
# coefs_reduced = az.summary(reduced_trace, kind='stats')
# print(coefs_reduced)
#
# # Plot posterior distributions
# az.plot_posterior(complete_trace, hdi_prob=0.95)
# az.plot_posterior(reduced_trace, hdi_prob=0.95)
#
# # Compare Bayesian models
# compare_dict = {
#     "full_model": complete_trace,
#     "restricted_model": reduced_trace
# }
#
# print(az.compare(compare_dict, scale='deviance'))
#




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


# Retrieve results from Bayesian Random Forest
print("Training bayesian_random_forest..")

rmse_brf, hyper, feat_num, feat_names = bayesian_random_forest(train, valid, test)
print('---------------')
print(rmse_brf)
print('---------------')
print(hyper)
print('---------------')
print(feat_num)
print('---------------')
print(feat_names)

model_results.append({
    "model": "Random Forest (Bayesian)",
    "rmse": rmse_brf,
    "details": hyper
})

# Retrieve results from Frequentist Extreme Random Forest
print("Training freq_ext_rf..")

rmse_ext, hyper, feat_num, feat_names = freq_ext_rf(train, valid, test)
print('---------------')
print(rmse_ext)
print('---------------')
print(hyper)
print('---------------')
print(feat_num)
print('---------------')
print(feat_names)

model_results.append({
    "model": "Extreme Random Forest (Frequentist)",
    "rmse": rmse_ext,
    "details": hyper
})

# Retrieve results from Bayesian Extreme Random Forest
print("Training Payes_ext_rf...")

rmse_bext, hyper, feat_num, feat_names = bayes_ext_rf(train, valid, test)
print('---------------')
print(rmse_bext)
print('---------------')
print(hyper)
print('---------------')
print(feat_num)
print('---------------')
print(feat_names)

model_results.append({
    "model": "Extreme Random Forest (Bayesian)",
    "rmse": rmse_bext,
    "details": hyper
})

# # Retrieve results from Artificial Neural Network
###not working
# if __name__ == '__main__':
#     best_run, best_model = optim.minimize(model=neural_network,
#                                           data=nn_data, algo=tpe.suggest,
#                                           max_evals=5, trials=Trials())
#
#     # Unbiased estimate of best model performance
#     X_train, y_train, X_valid, y_valid, X_test, y_test = nn_data()
#     print('Evalutation of best performing model:')
#     print(best_model.evaluate(X_valid, y_valid))
#     print('Best performing model chosen hyper-parameters:')
#     print(best_run)
#     print('Evalutation of best performing model for test set:')
#     print(best_model.evaluate(X_test, y_test))
# # Retrieve results from Bayesian Neural Network
# rmse = BNN(train, valid, test)

##new pytorch ANN
print("Training PyTorch ANN...")

ann_model, ann_val_rmse = train_ann(X_train, y_train, X_valid, y_valid)
print("Best Validation RMSE from ANN:", ann_val_rmse)

ann_test_rmse, ann_pred = evaluate_ann(ann_model, X_test, y_test)
print("Test RMSE from ANN:", ann_test_rmse)

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
#
# # XAI methods to interpret XGBoost outputs
# # SHAP values and plots for Global Interpretability
# shap_values = XAI_SHAP(model, test[0], 'global')
#
# # SHAP values and plots for Local Interpretability
# shap_values = XAI_SHAP(model, test[0], 'local', 10)
# shap_values = XAI_SHAP(model, test[0], 'local', 11)
# shap_values = XAI_SHAP(model, test[0], 'local', 12)
#
# # PDP and ICE plots for Global and Local interpretability
# XAI_PDP_ICE(model, test[0], 2, 5, ice=False)
# XAI_PDP_ICE(model, test[0], 11, 12, ice=False)
# XAI_PDP_ICE(model, test[0], 27, 28, ice=False)
# XAI_PDP_ICE(model, test[0], 2, 5, ice=True)
# XAI_PDP_ICE(model, test[0], 8, 9, ice=True)
# XAI_PDP_ICE(model, test[0], 11, 12, ice=True)
# XAI_PDP_ICE(model, test[0], 16, 17, ice=True)
# XAI_PDP_ICE(model, test[0], 71, 72, ice=True)
# XAI_PDP_ICE(model, test[0], 27, 28, ice=True)
