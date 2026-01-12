# import arviz
# import pandas as pd
#
# print(arviz.__version__)
# print(arviz.__file__)

# Load required libraries and custom functions
import os

from Functions.ml_models import *
from Functions.dl_models import *
from Functions.cv_functions import *
from Functions.nlp_functions import *
from Functions.general_functions import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # # Retrieving data using our defined function
# #
listings = pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\POI\2025_09_ListingPOIwithin800m.csv')
print(listings.shape)
print(listings.columns)

##second option
# import json
# print(listings.shape)
# print(listings['room_type'].unique().tolist())
# for rt,rt_list in zip(['home','room'],[['Entire home/apt'.lower()],['Private room'.lower(),'Shared room'.lower()]]):
#     listings_bytypes=listings[listings['room_type'].str.lower().isin(rt_list)]['id'].unique().tolist()
#     print(len(listings_bytypes))
#
#     with open(rf"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\listings_{rt}_2024_09.geojson", "w") as f:
#         json.dump(listings_bytypes, f, indent=2)
##second option


# Removing inactive users
listings = listings.drop(listings[listings['host_since'].isnull()].index)

listings = listings.drop(
    listings[listings['maximum_nights_avg_ntm'].isnull()].index)

listings = listings.reset_index(drop=True)

# Feature Engineering and Data Cleaning
# Removing dollar symbol from price and taking logarithms
print(listings[['price']].head(3))
print(listings.shape)
##old
# listings=listings[listings['price'].notnull()]
# price = [y[1:] for y in listings['price']]
# price = [int(float(y.replace(',', ''))) for y in price]
# p90_reviews = listings["price"].quantile(0.95)
# listings = listings[listings["price"] <= p90_reviews]
# listings['price'] = [np.log(y) if y != 0 else 0 for y in price]
# print(listings.shape)

print(listings.shape)
listings = listings[listings['price'].notnull()]
listings['price'] = (
    listings['price']
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
    .astype(float)
)
p95 = listings['price'].quantile(0.95)
listings = listings[listings['price'] <= p95]
listings['price'] = listings['price'].apply(lambda x: np.log(x) if x > 0 else 0)
print(listings.shape)


##ignore extreme large number of reviews
# listings = listings[listings['number_of_reviews'].notnull()]
# p90_reviews = listings["number_of_reviews"].quantile(0.95)
# listings = listings[listings["number_of_reviews"] <= p90_reviews]
# listings['number_of_reviews'] = [np.log(y) if y != 0 else 0 for y in p90_reviews]
# print(listings.shape)

listings = listings[listings['number_of_reviews'].notnull()]
p95 = listings["number_of_reviews"].quantile(0.95)
listings = listings[listings["number_of_reviews"] <= p95]
listings['number_of_reviews'] = listings['number_of_reviews'].apply(
    lambda y: np.log(y) if y > 0 else 0
)
print(listings.shape)

# Encoding neighborhood clusters
groups = 'neighbourhood_group_cleansed'
listings['neighborhood'] = listings[groups].astype('category').cat.codes

# Encoding specific neighborhoods
groups = 'neighbourhood_cleansed'
listings['neighborhood_group'] = listings[groups].astype('category').cat.codes

# Imputing empty reviews and creating new feature identifying non-reviewed listings
listings['inactive'] = listings['reviews_per_month'].replace(np.nan, 'inact')
listings['inactive'] = [0 if i != 'inact' else 1 for i in listings['inactive']]
listings['reviews_month'] = listings['reviews_per_month'].replace(np.nan, 0)
listings = listings.drop('reviews_per_month', axis=1)

# Encoding room type
listings['room_type'] = listings['room_type'].replace(
    ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'],
    [1, 2, 3, 4])

# Encoding host response time and creating new feature identifying non-respondent
listings['host_response_time'] = listings['host_response_time'].replace(
    [np.nan, 'within an hour', 'within a few hours', 'within a day', 'a few days or more'],
    [0, 1, 2, 3, 4])

# Detecting hosts that do not respond
listings['responds'] = listings['host_response_rate'].replace(
    [np.nan, '0%'], [0, 0])

listings['responds'] = [1 if i != 0 else 0 for i in listings['responds']]

# Removing '%' symbol from response rate and coercing it into an integer
for i in ['host_response_rate', 'host_acceptance_rate']:
    rate = listings[i].replace(np.nan, 0)
    rate = [x if x == 0 else x.replace('%', '') for x in rate]
    rate = [int(j) for j in rate]
    listings[i] = rate

# One-hot encoding several features
features = [
    'has_availability', 'instant_bookable', 'host_has_profile_pic',
    'host_is_superhost', 'host_identity_verified'
]

for i in features:
    listings[i] = listings[i].apply(lambda x: 1 if x == 't' else 0)

# Calculating days since different events
print(listings.shape)
listings=listings[listings['last_review'].notnull() & listings['first_review'].notnull() & listings['host_since'].notnull()]
print(listings.shape)
for j in ['last_review', 'first_review', 'host_since']:
    listings[j] = calculate_days(listings['last_scraped'], listings[j])

# Identifying number of bathrooms
print(listings['bathrooms_text'].unique().tolist())
listings['bathrooms_text'] = listings['bathrooms_text'].replace(
    ['Private half-bath', np.nan, 'Half-bath', 'Shared half-bath'],
    ['1 private', '0 baths', '1 private', '1 private'])

listings['bathrooms'] = [int(j[0]) for j in listings['bathrooms_text']]
listings = listings.drop('bathrooms_text', axis=1)

# Normalized space for longitude and latitude
lat, lon = listings['latitude'], listings['longitude']
listings['geo_x'] = np.multiply(np.cos(lat), np.cos(lon))
listings['geo_y'] = np.multiply(np.cos(lat), np.sin(lon))
listings['geo_z'] = np.sin(lat)
listings = listings.drop(['latitude', 'longitude'], axis=1)

# Check whether host supplied information about listing
listings['description'] = listings['description'].replace(np.nan, 0)
listings['description'] = [1 if i != 0 else 0 for i in listings['description']]

# Encoding
print(listings['property_type'].unique().tolist())
listings['property'] = listings['property_type'].astype('category').cat.codes

# Selecting some amenities
print(listings['amenities'].unique().tolist())
one_hot_encoder('tv', listings, 'amenities', 'television')
one_hot_encoder('netflix', listings, 'amenities', 'amazon')
one_hot_encoder('gym', listings, 'amenities', 'gym')
one_hot_encoder('elevator', listings, 'amenities', 'lift')
one_hot_encoder('fridge', listings, 'amenities', 'refrigerator')
one_hot_encoder('heating', listings, 'amenities', 'heating')
one_hot_encoder('hair_dryer', listings, 'amenities', 'hair dryer')
one_hot_encoder('air_conditioning', listings, 'amenities', 'air conditioning')
one_hot_encoder('hot_tub', listings, 'amenities', 'hot tub')
one_hot_encoder('oven', listings, 'amenities', 'oven')
one_hot_encoder('bbq', listings, 'amenities', 'barbecue')
one_hot_encoder('security cameras', listings, 'amenities', 'camera')
one_hot_encoder('workspace', listings, 'amenities', 'workspace')
one_hot_encoder('coffee', listings, 'amenities', 'coffee maker')
one_hot_encoder('backyard', listings, 'amenities', 'backyard')
one_hot_encoder('outdoor_dining', listings, 'amenities', 'outdoor dining')
one_hot_encoder('greets', listings, 'amenities', 'host greets')
one_hot_encoder('pool', listings, 'amenities', 'pool')
one_hot_encoder('beachfront', listings, 'amenities', 'beach view')
one_hot_encoder('patio', listings, 'amenities', 'balcony')
one_hot_encoder('luggage', listings, 'amenities', 'luggage dropoff')
one_hot_encoder('furniture', listings, 'amenities', 'outdoor furniture')


# Get hosts' gender
#amended
d = gender.Detector()

# Apply name gender detector
listings['name_gender'] = listings['host_name'].apply(lambda x: d.get_gender(x) if isinstance(x, str) else 'unknown')

# Map gender-guesser results to numeric codes
gender_map = {
    'female': 0, 'mostly_female': 0,
    'male': 1, 'mostly_male': 1,
    'unknown': np.nan, 'andy': np.nan,
    None: np.nan
}
listings['name_gender_code'] = listings['name_gender'].map(gender_map)


# ==========================================================
# 2. Run CV model ONLY for hosts still missing gender
# ==========================================================
missing_ids = listings[listings['name_gender_code'].isna()]['id']
print(len(missing_ids))

cv_data = cv_model(listings[listings['id'].isin(missing_ids)])
print(listings[listings['id'].isin(missing_ids)].head())
print(listings[listings['id'].isin(missing_ids)].shape)

# Keep only valid CV predictions
cv_data['cv_gender'] = cv_data['cv_gender'].apply(lambda x: np.nan if x not in ['Woman', 'Man'] else x)
cv_data['cv_gender'] = cv_data['cv_gender'].replace(['Woman', 'Man'], [0, 1])
print(cv_data.head())
print(cv_data.shape)

# cv_data.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\cvout_2024_09.csv')
#
# cv_data=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\cvout_2024_09.csv')
# print(cv_data.shape)
#
# # Merge CV predictions BACK into listings (ONLY missing ones)
listings = listings.merge(cv_data[['id', 'cv_gender']], on='id', how='left')

# Final gender: use name_gender unless missing → falls back to cv_gender
listings['gender_final'] = listings['name_gender_code']
listings.loc[listings['gender_final'].isna(), 'gender_final'] = listings['cv_gender']

# ==========================================================
# 3. Manual replacements (your list of female names)
# ==========================================================
print(listings[listings['gender_final'].isnull()]['host_name'].unique().tolist())

female = [
    'Abrianna', 'Brinda', 'Denisse', 'Tytyana', 'Susian', 'Diba',
    'Feven', 'Flor Arely', 'Dipti', 'Shaquetta', 'Jonitha'
]
fix_ids = listings[listings['host_name'].isin(female)].index
listings.loc[fix_ids, 'gender_final'] = 0  # female = 0

# ==========================================================
# 4. Impute remaining NaNs using mode
# ==========================================================
mode_gender = stats.mode(listings['gender_final'], nan_policy='omit')[0][0]
listings['gender_final'] = listings['gender_final'].fillna(mode_gender)

# ==========================================================
# 5. Create categorical version
# ==========================================================
print(listings[listings['gender_final'].isnull()]['host_name'].unique().tolist())

listings['gender_final'] = listings['gender_final'].astype('category')

listings.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_lst_clean_2022_09_p95.csv')

# ##explore price and number of review range
for yr in [2022,2023,2024, 2025]:
    df=pd.read_csv(rf'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_lst_clean_{yr}_09_p95.csv',index_col=[0])
    print(df.shape)
    print(df.columns)
    print(df['price'].describe())
    print(df['number_of_reviews'].describe())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Price histogram
    axes[0].hist(df["price"], bins=40, color="steelblue", alpha=0.7)
    axes[0].set_title("Distribution of Price")
    axes[0].set_xlabel("Price")
    axes[0].set_ylabel("Count")

    # Number of reviews histogram
    axes[1].hist(df["number_of_reviews"], bins=40, color="darkorange", alpha=0.7)
    axes[1].set_title("Distribution of Number of Reviews")
    axes[1].set_xlabel("Number of Reviews")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()




# #clean reviews
# Reading reviews and ditching out non-latin symbols
os.chdir(
    r'C:\Users\yw30f\OneDrive - University of Glasgow\Airbnb\regulations\InsideAirbnb\data\insideAirbnbScraping_14102025\Edinburgh\hidden')

reviews = read_url('2024-09-13_reviews.csv.gz')
print(reviews.columns)
print(reviews.shape)
reviews['comments'] = only_latin(reviews['comments'])
langs = [classify(i)[0] for i in reviews['comments']]
ids = [j for j in range(len(langs)) if langs[j] != 'en']
reviews = reviews.drop(labels=ids, axis=0)
reviews = reviews.reset_index(drop=True)

# Predict sentiment
reviews['sentiment'] = [sentiment_vader(clean_text(i)) for i in reviews['comments']]
reviews.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_reviews_2024_09.csv')

# ##merge step
# # ##preprocessed read
reviews= pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_reviews_2024_09.csv')
listings= pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_lst_clean_2024_09_P95.csv')


# # Average sentiment for each listing
sent_avg = reviews.groupby('listing_id', as_index=False)['sentiment'].mean()
sent_median = reviews.groupby('listing_id', as_index=False)['sentiment'].median()
sent_mode = reviews.groupby('listing_id', as_index=False)['sentiment'].agg(
    lambda x: x.value_counts().index[0])

# Set up columns for later SQL join
sent_avg.columns = ['id', 'sent_mean']
sent_median.columns = ['id', 'sent_median']
sent_mode.columns = ['id', 'sent_mode']

# Add average, median and mode sentiment to original dataset
listings = listings.merge(sent_median, on='id', how='left')
print(listings.shape)
print(listings[listings['sent_median'].isnull()].shape)
listings = listings.merge(sent_avg, on='id', how='left')
listings = listings.merge(sent_mode, on='id', how='left')

# Encode mode
listings['sent_mode'] = listings['sent_mode'].astype('category')

# Identify non-commented listings
listings['no_review'] = listings['sent_mean'].fillna('nan')
listings['no_review'] = [0 if i == 'nan' else 1 for i in listings['no_review']]

# Drop unused features
unused = [
    'id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'picture_url',
    'neighborhood_overview', 'host_id', 'host_url', 'host_name', 'host_location',
    'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
    'host_total_listings_count', 'host_verifications', 'neighbourhood', 'amenities',
    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
    'maximum_maximum_nights', 'calendar_updated', 'availability_60', 'license',
    'calendar_last_scraped', 'property_type', 'neighbourhood_group_cleansed',
    'neighbourhood_cleansed'
]

listings = listings.drop(unused, axis=1)

# Correctly encoding categorical features
categorical = [
    'property', 'room_type', 'neighborhood', 'neighborhood_group',
    'host_response_time', 'sent_mode', 'cv_gender', 'gender_final'
]

for i in categorical:
    listings[i] = listings[i].astype('category')

# Remove gender from CV model
removeGenderCols=[c for c in listings.columns if '_gender' in c ]
listings = listings.drop(removeGenderCols, axis=1)
print(listings.columns)
listings.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2024_09_P95.csv')

#merge poi and llm
#
# listings=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2025_09_P95.csv')
# print(listings.shape)
# print(listings.columns)
# listings['listing_id']=listings['listing_id'].astype(int)
#
# df_poi_llm=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\listingout_merge_locationcount_onlyIdCount_tidy_fewCol_2025_09_v2.csv',index_col=[0])
# print(df_poi_llm.shape)
# print(df_poi_llm.columns)
# df_poi_llm=df_poi_llm[['id','n_specific',	'n_total',	'n_generalParent']]
# print(df_poi_llm.shape)
# print(df_poi_llm.columns)
# df_poi_llm['id']=df_poi_llm['id'].astype(int)
#
# listings=listings.merge(df_poi_llm,left_on='listing_id',right_on='id',how='inner')
# print(listings.shape)
# print(listings.columns)
# print(listings[listings['poi_within_800m'].isnull()].shape)
# print(listings[listings['n_total'].isnull()].shape)
# listings=listings[listings['n_total'].notnull()]
# print(listings.shape)
# print(listings.columns)
#
# cols=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'source', 'description',
#       'minimum_nights_avg_ntm',  'maximum_nights_avg_ntm','has_availability', 'availability_90', 'availability_365', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
#  'availability_eoy', 'number_of_reviews_ly', 'estimated_occupancy_l365d', 'estimated_revenue_l365d',
# 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',       'review_scores_communication', 'review_scores_location',       'review_scores_value',
# 'calculated_host_listings_count_entire_homes',       'calculated_host_listings_count_private_rooms',       'calculated_host_listings_count_shared_rooms',
# 'geometry',       'buffer_800m','neighborhood_group','geo_x', 'geo_y', 'geo_z',
# 'sent_median', 'sent_mode', 'no_review','responds', 'neighborhood', 'review_scores_rating','id'
#  ]
#
# df_lst=listings[[c for c in listings.columns if c not in cols]]
# print(df_lst.shape)
# print(df_lst.columns)
# print(len(df_lst.columns.tolist()))
#
# df_lst.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\P95_feature1_202509.csv')


# # ##tidy up non useful columns
# df_lst=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\P95_feature1_202209.csv',index_col=[0])
# print(df_lst.shape)
# print(df_lst.columns)
# print(len(df_lst.columns.tolist()))
#
# cols=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'source', 'description',
#       'minimum_nights_avg_ntm',  'maximum_nights_avg_ntm','has_availability', 'availability_90', 'availability_365', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
#  'availability_eoy', 'number_of_reviews_ly', 'estimated_occupancy_l365d', 'estimated_revenue_l365d',
# 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',       'review_scores_communication', 'review_scores_location',       'review_scores_value',
# 'calculated_host_listings_count_entire_homes',       'calculated_host_listings_count_private_rooms',       'calculated_host_listings_count_shared_rooms',
# 'geometry',       'buffer_800m','neighborhood_group','geo_x', 'geo_y', 'geo_z',
# 'sent_median', 'sent_mode', 'no_review','responds', 'neighborhood', 'review_scores_rating'
#  ]
#
# df_lst=df_lst[[c for c in df_lst.columns if c not in cols]]
# print(df_lst.shape)
# print(df_lst.columns)
# print(len(df_lst.columns.tolist()))
#
# # print(df_lst[df_lst['review_scores_rating'].isnull()].shape)
#
# df_lst.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\tuneXgboost_feature2_202209.csv')
#
# df_pre=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\tuneXgboost_feature4.csv',index_col=[0])
# print(df_lst.shape)
# print(df_lst.columns)
# print(len(df_lst.columns.tolist()))
#
# df_pre=df_pre[[c for c in df_pre.columns if c not in cols]]
# print(df_pre.shape)
# print(df_pre.columns)
# print(len(df_pre.columns.tolist()))
#
# #adding review_score_rating
# #found too many missing values#decide to drop
# df_add_23=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2023_09.csv',index_col=[0])
# print(df_add_23.columns)
# df_add_23=df_add_23[['listing_id','review_scores_rating']]
#
# df_pre=df_pre.merge(df_add_23,on='listing_id',how='left')
# print(df_pre)
# print(df_pre.shape)
# print(df_pre.columns)
# print(len(df_pre.columns.tolist()))
# print(df_pre[df_pre['review_scores_rating'].isnull()].shape)
#
# df_pre.to_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\tuneXgboost_feature5.csv')
# #adding review_score_rating
# #found too many missing values#decide to drop


# ##report summary of variables
# df=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\P95_feature1_202509.csv', index_col=[0])
# print(df.columns)
# print(df.shape)




