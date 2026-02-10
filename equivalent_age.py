import pandas as pd
import numpy as np

def calculate_equ_age(data):
    """
    Calculates the equivalent age using linear interpolation based on DALYs rates.
    """

    subset = data[data['closest_to_global65'] == 2].copy()
    
    if len(subset) < 2:
        return np.nan

    subset = subset.reset_index(drop=True)

    rate_global65 = data['global65'].iloc[0]
    
    rate_x1 = subset['DALYs_rate'].iloc[0]
    rate_x2 = subset['DALYs_rate'].iloc[1]
    age_x1 = subset['age_median'].iloc[0]
    age_x2 = subset['age_median'].iloc[1]
    
    if rate_x2 - rate_x1 == 0:
        return np.nan
        
    equ_age = age_x1 + (rate_global65 - rate_x1) * (age_x2 - age_x1) / (rate_x2 - rate_x1)

    return equ_age

# Load data
data = pd.read_csv("equ_age_sample_data.csv")

#### 1. Calculate age-related burden rate for each location and age group ####
grouped = data.groupby(['age_name', 'location_id'])

# Calculate sums
df_agg = grouped.agg({
    'val': 'sum',
    'val_pop': 'sum',
    'lower': 'sum',
    'upper': 'sum',
    'age_id': 'first',      
    'age_median': 'first'   
}).reset_index()

# Calculate rates
df_agg['DALYs_rate'] = df_agg['val'] / df_agg['val_pop']
df_agg['DALYs_rate_lower'] = df_agg['lower'] / df_agg['val_pop']
df_agg['DALYs_rate_upper'] = df_agg['upper'] / df_agg['val_pop']

df = df_agg[['location_id', 'age_id', 'age_name', 'age_median', 'DALYs_rate', 'DALYs_rate_lower', 'DALYs_rate_upper']].copy()

# Calculate global reference values (Location ID 1, ages 60-64 and 65-69)
loc1_60_64 = df[(df['location_id'] == 1) & (df['age_name'] == "60-64 years")]
loc1_65_69 = df[(df['location_id'] == 1) & (df['age_name'] == "65-69 years")]


rate_60_64 = loc1_60_64['DALYs_rate'].values[0]
rate_65_69 = loc1_65_69['DALYs_rate'].values[0]

lower_60_64 = loc1_60_64['DALYs_rate_lower'].values[0]
lower_65_69 = loc1_65_69['DALYs_rate_lower'].values[0]

upper_60_64 = loc1_60_64['DALYs_rate_upper'].values[0]
upper_65_69 = loc1_65_69['DALYs_rate_upper'].values[0]

global65 = (rate_60_64 + rate_65_69) / 2
global65_lower = (lower_60_64 + lower_65_69) / 2
global65_upper = (upper_60_64 + upper_65_69) / 2

df['global65'] = global65
df['global65_lower'] = global65_lower
df['global65_upper'] = global65_upper

#### 2. Calculate equivalent age Step 1: Identify adjacent ages ####

df['abs_diff'] = (df['DALYs_rate'] - df['global65']).abs()
min_diffs = df.groupby('location_id')['abs_diff'].transform('min')
df['closest_to_global65'] = np.where(df['abs_diff'] == min_diffs, 1, 0)
df = df.sort_values(by=['location_id', 'age_median'])
g = df.groupby('location_id')['closest_to_global65']
df['lag_closest'] = g.shift(1)
df['lead_closest'] = g.shift(-1)

conditions = [
    (df['lag_closest'] == 1),
    (df['lead_closest'] == 1)
]
choices = [2, 2]

mask_neighbor = (df['lag_closest'] == 1) | (df['lead_closest'] == 1)
df.loc[mask_neighbor, 'closest_to_global65'] = 2

df = df.drop(columns=['abs_diff', 'lag_closest', 'lead_closest'])

#### 2. Calculate equivalent age ####
results_list = []
countries = df['location_id'].unique()

for country in countries:
    country_data = df[df['location_id'] == country]
    equ_age = calculate_equ_age(country_data)
    
    results_list.append({
        'location_id': country,
        'equ_age': equ_age
    })

results = pd.DataFrame(results_list)


results.to_csv("equ_age_result.csv", index=False)
