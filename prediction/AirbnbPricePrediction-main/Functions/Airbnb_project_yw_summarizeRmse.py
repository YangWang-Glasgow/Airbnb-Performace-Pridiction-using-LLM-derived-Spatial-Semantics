import os
import pandas as pd

# Root directory containing your rf_experiments_* folders
root_dir = r"C:\Users\yw30f\OneDrive - University of Glasgow\LLM\data"

# Will store all results here
all_results = []

# Loop over all subfolders beginning with "rf_experiments"
for yr in [2022,2023,2024,2025]:
# for folder in os.listdir(root_dir):
#     if folder.startswith("rf_experiments") and folder.endswith(""):
        folder=f'xgb_experiments_reviews_{yr}'
        experiment_path = os.path.join(root_dir, folder, "rmse_comparisons")

        if not os.path.exists(experiment_path):
            continue

        print(f"Scanning {experiment_path}")

        # Loop through all results inside rmse_comparisons
        for file in os.listdir(experiment_path):
            if file.endswith(".csv") and file.startswith("experiment_results"):

                file_path = os.path.join(experiment_path, file)

                # Load CSV
                df = pd.read_csv(file_path)

                # Add metadata
                df["experiment_folder"] = folder
                df["scenario"] = file.replace("experiment_results_", "").replace(".csv", "")

                all_results.append(df)

# Combine into one dataframe
if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)

    unique_cols = ["feature", "geo", "poi", "experiment_folder"]

    clean_df = (
        summary_df.sort_values("rmse")
        .drop_duplicates(subset=unique_cols, keep="first")
    )


    # Save final summary
    summary_path = os.path.join(f"{root_dir}/xgb_summary", "xgb_experiments_summary_reviews_all_unique_p95_noDrop.csv")
    clean_df.to_csv(summary_path, index=False)

    print("\n==============================")
    print(" Master Summary Created!")
    print("==============================")
    print(f"Saved → {summary_path}")
    print("\nPreview:")
    print(summary_df.head())

else:
    print("⚠ No experiment result CSV files found.")


df=clean_df.copy()
# df is your full combined results table
# columns: feature, geo, poi, rmse, experiment_folder, scenario

import pandas as pd

# df is your combined dataframe

# Convert TRUE/FALSE strings → booleans
df["geo"] = df["geo"].astype(str).str.upper().map({"FALSE": False, "TRUE": True})
df["poi"] = df["poi"].astype(str).str.upper().map({"FALSE": False, "TRUE": True})

# ---------------------------------------------------------
# Desired sort order
# ---------------------------------------------------------
feature_order = ["none", "n_total", "n_generalParent"]
bool_order = {False: 0, True: 1}

def sort_key(row):
    return (
        feature_order.index(row["feature"]),
        bool_order[row["geo"]],
        bool_order[row["poi"]],
    )

# ---------------------------------------------------------
# Compute RMSE improvement within each experiment folder
# ---------------------------------------------------------
def compute_improvement(group):
    baseline = group[
        (group["feature"] == "none") &
        (group["geo"] == False) &
        (group["poi"] == False)
    ]["rmse"]

    if baseline.empty:
        group["rmse_improvement"] = None
        return group

    baseline_rmse = baseline.iloc[0]
    group["rmse_improvement"] = group["rmse"] - baseline_rmse
    return group

df = df.groupby("experiment_folder", group_keys=False).apply(compute_improvement)

# ---------------------------------------------------------
# Final sorting using custom sort_key
# ---------------------------------------------------------
df["sort_key"] = df.apply(sort_key, axis=1)

df = df.sort_values(
    by=["experiment_folder", "sort_key"]
).drop(columns="sort_key")

print(df)
summary_path = os.path.join(f"{root_dir}/xgb_summary", "xgb_experiments_summary_reviews_all_unique_p95_noDrop_sorted.csv")
df.to_csv(summary_path, index=False)
