import pandas as pd
import numpy as np

def calculate_daly_rate_age(dalys, pop):
    """Calculate DALY rate for each age group"""
    return np.sum(dalys) / pop

def age_adjust_direct(daly_rates, std_pop):
    """Calculate age-adjusted rate"""
    return np.sum(daly_rates * std_pop) / np.sum(std_pop)


# read data
df = pd.read_csv("GBD_sample_data.csv")

# Create an empty data frame to store all the results
columns = ['year', 'location_id', 'DALYs', 'adj_rate', ]
results = []

years = df['year'].unique()
locations = df['location_id'].unique()

for year in years:
    for location in locations:

        df_subset = df[(df['year'] == year) & (df['location_id'] == location)]

        if df_subset.empty:
            continue

        grouped = df_subset.groupby('age_id').agg({
            'val': 'sum',  
            'val_pop': 'first',  
            'age_standard_pop_proportion': 'first'  
        }).reset_index()

        # age-related DALYs 
        DALYs = df_subset['val'].sum()

        # age_standardised_DALYs
        grouped['daly_rate_age'] = grouped.apply(lambda row: calculate_daly_rate_age(row['val'], row['val_pop']), axis=1)
        adj_rate = age_adjust_direct(grouped['daly_rate_age'], grouped['age_standard_pop_proportion'])

        results.append({
            'year': year,
            'location_id': location,
            'DALYs': "{:.1f}".format(round(DALYs / 1000, 1)), # thousand
            'adj_rate': "{:.1f}".format(round(adj_rate * 1000, 1)),
        })


result_df = pd.DataFrame(results)
result_df.to_csv("DALYs_results.csv", index=False)