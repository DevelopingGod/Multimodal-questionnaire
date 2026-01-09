import pandas as pd
import os

# 1. SETUP
input_file = "corrected_mental_health_dataset.csv" 
# Fallback if corrected version doesn't exist
if not os.path.exists(input_file):
    input_file = "synthetic_college_mental_health_dataset_v3_with_severity.csv"

output_file = "final_dataset_no_sleep.csv"

print(f"Reading from: {input_file}")
df = pd.read_csv(input_file)

# 2. SEPARATE AND REMOVE SLEEP FEATURES
# Identify all Question columns
all_q_cols = [c for c in df.columns if "Q" in c]

# Identify Sleep/Confound columns (CONF_Q...)
sleep_cols = [c for c in all_q_cols if "CONF" in c]

# Create list of features to KEEP (Excluding sleep)
final_feature_cols = [c for c in all_q_cols if c not in sleep_cols]

print(f"\nFound {len(sleep_cols)} Sleep/Confound columns: {sleep_cols}")
print(f"Removing them...")

# Drop the columns from the dataframe
df_clean = df.drop(columns=sleep_cols)

# 3. SAVE
df_clean.to_csv(output_file, index=False)

print(f"\n✅ Success! Cleaned dataset saved as: '{output_file}'")
print(f"Remaining Feature Count: {len(final_feature_cols)}")
print(f"Columns: {final_feature_cols}")