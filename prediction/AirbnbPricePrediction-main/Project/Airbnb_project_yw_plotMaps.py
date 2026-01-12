import geopandas as gpd


# Load required libraries and custom functions
import os
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # # # Retrieving data using our defined function
# #


listings=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2025_09_P95.csv')
listings['listing_id']=listings['listing_id'].astype(int)
listings=listings[['listing_id','poi_within_800m']]
print(listings.shape)
print(listings.columns)


df_poi_llm=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\test\edinburgh\test\listingout_merge_locationcount_onlyIdCount_tidy_fewCol_2025_09_v2.csv',index_col=[0])

df_poi_llm=df_poi_llm[['id','n_specific',	'n_total',	'n_generalParent']]
print(df_poi_llm.shape)
print(df_poi_llm.columns)
df_poi_llm['id']=df_poi_llm['id'].astype(int)

listings=listings.merge(df_poi_llm,left_on='listing_id',right_on='id',how='inner')
print(listings.shape)
print(listings.columns)
print(listings[listings['poi_within_800m'].isnull()].shape)
print(listings[listings['n_total'].isnull()].shape)
listings=listings[listings['n_total'].notnull()]
print(listings.shape)
print(listings.columns)



listings_location = pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\POI\2025_09_ListingPOIwithin800m.csv')

listings_location=listings_location[['listing_id','latitude','longitude','number_of_reviews']]
print(listings_location.shape)
print(listings_location.columns)
listings_location['listing_id']=listings_location['listing_id'].astype(int)

listings_location=listings_location.merge(listings,left_on='listing_id',right_on='listing_id',how='left')
print(listings_location.shape)
print(listings_location.columns)
print(listings_location[listings_location['poi_within_800m'].isnull()].shape)
print(listings_location[listings_location['n_total'].isnull()].shape)


listings_location=listings_location[(listings_location['n_total'].notnull()) & (listings_location['poi_within_800m'].notnull())]
print(listings_location.shape)
print(listings_location.columns)
print(listings_location['poi_within_800m'].describe())
print(listings_location['n_total'].describe())
#
# print(listings_location[listings_location['poi_within_800m']>=2000]['listing_id'].tolist())
# lst=listings_location[listings_location['poi_within_800m']>=2000]['listing_id'].tolist()
# #
# # listings_location=pd.read_csv(r'C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data\modelfit_listings_clean_2025_09_P95.csv')
# print(listings_location.columns)
# print(listings_location[listings_location['listing_id'].isin(lst)][['latitude','longitude']])

##add original pois
import json
import geopandas as gpd
from h3 import h3
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import contextily as cx

with open(r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\POI\CombinedPOIs.geojson") as f:
    data = json.load(f)

gdf_poi = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:27700").to_crs(epsg=4326)
print(gdf_poi.shape)
print(gdf_poi.head())
print(gdf_poi.crs)



# 1. Create GeoDataFrame (KEEP EPSG:4326)
gdf_listings  = gpd.GeoDataFrame(
    listings_location,
    geometry=gpd.points_from_xy(
        listings_location.longitude,
        listings_location.latitude
    ),
    crs="EPSG:4326"
)

# 2. Compute H3 (must be 4326!)
h3_level = 9

def geom_to_h3(geom):
    return h3.geo_to_h3(geom.y, geom.x, h3_level)

gdf_listings ["h3"] = gdf_listings.geometry.apply(geom_to_h3)
gdf_poi["h3"] = gdf_poi.geometry.apply(geom_to_h3)

print("Unique H3 cells (listings):", gdf_listings["h3"].nunique())
print("Unique H3 cells (POIs):", gdf_poi["h3"].nunique())

poi_agg = (
    gdf_poi
    .groupby("h3")
    .size()
    .reset_index(name="poi_count")
)
print(poi_agg.head())


# 3. Count listings per H3
listings_agg = (
    gdf_listings
    .groupby("h3")
    .agg(
        listing_count=("listing_id", "count"),
        n_total_sum=("n_total", "sum"),
        poi_within_800m_sum=("poi_within_800m", "sum"),
        total_reviews=("number_of_reviews", "sum")
    )
    .reset_index()
)
print(listings_agg.head())

counts = listings_agg.merge(
    poi_agg,
    on="h3",
    how="left"
)
counts["poi_count"] = counts["poi_count"].fillna(0).astype(int)
print(counts.head())



# 4. Convert H3 to polygons
def h3_to_polygon(h3_index):
    return Polygon(
        h3.h3_to_geo_boundary(h3_index, geo_json=True)
    )

counts["geometry"] = counts["h3"].apply(h3_to_polygon)



gdf_h3 = gpd.GeoDataFrame(
    counts,
    geometry="geometry",
    crs="EPSG:4326"
)

print(gdf_h3.head())

# 5. Reproject for plotting (Web Mercator!)
gdf_h3_3857 = gdf_h3.to_crs(epsg=3857)


##load background shape
msoa = gpd.read_file(
    r"C:\Users\yw30f\OneDrive - University of Glasgow\Airbnb\nikeScraping\SIMD\Cities\Edinburgh\msoa_tenureRatio.geojson"
)

print(msoa.shape)
print(msoa.crs)
print(msoa.head())
msoa_3857 = msoa.to_crs(epsg=3857)


# # 6. Plot
# fig, ax = plt.subplots(figsize=(8, 8))
#
# gdf_h3_3857.plot(
#     ax=ax,
#     column="listing_count",
#     cmap="OrRd",
#     linewidth=0.3,
#     edgecolor="grey",
#     legend=True
# )
#
# cx.add_basemap(
#     ax,
#     source=cx.providers.CartoDB.Positron
# )
#
# ax.set_axis_off()
# ax.set_title("Listings per H3 cell", fontsize=12)
#
# plt.tight_layout()
# plt.show()

fields = {
    "listing_count": "Listings",
    "total_reviews": "Total reviews",
    "n_total_sum": "Spatial semantics",
    "poi_count": "POIs"
}

cmaps = ["OrRd", "Blues", "Greens", "Purples"]


fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(9, 9),
    constrained_layout=True
)

axes = axes.flatten()

for i, (ax, (col, title)) in enumerate(zip(axes, fields.items())):
    # MSOA / neighbourhood boundaries
    msoa_3857.boundary.plot(
        ax=ax,
        linewidth=0.3,
        edgecolor="black",
        alpha=0.2
    )

    # H3 hexes with different colour maps
    gdf_h3_3857.plot(
        ax=ax,
        column=col,
        cmap=cmaps[i % len(cmaps)],  # cycle safely
        linewidth=0.15,
        edgecolor="grey",
        legend=True
    )

    # Basemap
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.Positron,
        attribution=False
    )

    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.set_ylabel("")

plt.suptitle(
    "Key Variables (H3 = 1.11 km$^2$)",
    fontsize=14
)

plt.show()


