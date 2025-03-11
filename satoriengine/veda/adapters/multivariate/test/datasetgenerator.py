import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_datasets(num_rows=15000):
    """
    Generate three datasets:
    1. Original pattern extended
    2. Correlated dataset
    3. Non-correlated dataset
    
    Each dataset will have at least num_rows rows.
    """
    # Define the hourly pattern observed in the data
    hourly_avg = {
        0: 120, 1: 112, 2: 105, 3: 96, 4: 92, 5: 93, 
        6: 114, 7: 134, 8: 155, 9: 169, 10: 184, 11: 192,
        12: 188, 13: 179, 14: 174, 15: 182, 16: 194, 17: 215,
        18: 234, 19: 228, 20: 220, 21: 200, 22: 175, 23: 144
    }
    
    # Set a start date
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    
    # Calculate how many days we need to generate
    days_needed = (num_rows // 24) + 1  # Each day has 24 hours
    
    # Generate timestamps, values, and IDs
    timestamps = []
    values = []
    ids = []
    
    # Daily variation factor (some days might be busier or slower)
    daily_variations = {}
    
    for day in range(days_needed):
        # Create a daily variation factor between 0.9 and 1.1
        # This adds some natural variation between days
        daily_var = random.uniform(0.9, 1.1)
        
        # Add weekly patterns (weekends vs weekdays)
        current_date = start_date + timedelta(days=day)
        weekday = current_date.weekday()
        
        # Weekend adjustment (less activity on weekends)
        weekend_factor = 0.85 if weekday >= 5 else 1.0
        
        # Store the daily variation
        date_str = current_date.strftime('%Y-%m-%d')
        daily_variations[date_str] = daily_var * weekend_factor
        
        for hour in range(24):
            # Calculate timestamp
            timestamp = start_date + timedelta(days=day, hours=hour)
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add some noise to the hourly pattern (±10%)
            noise = random.uniform(0.9, 1.1)
            value = int(hourly_avg[hour] * daily_var * weekend_factor * noise)
            
            # Generate ID in the format YYYYMMDDHHXX
            id_str = timestamp.strftime('%Y%m%d%H') + '01'  # Last 2 digits fixed to 01
            
            timestamps.append(timestamp_str)
            values.append(value)
            ids.append(id_str)
    
    # Create the main dataset
    main_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'id': ids
    })
    
    # Create a correlated dataset (correlation ~0.8)
    # We'll create a dataset that follows a similar pattern but with more noise
    correlated_values = []
    for i, value in enumerate(values):
        timestamp = timestamps[i]
        date = timestamp.split()[0]
        hour = int(timestamp.split()[1].split(':')[0])
        
        # Base on the original value but add significant noise
        base = value * 0.8
        noise = random.uniform(0.7, 1.3)
        new_value = int(base + (base * noise * 0.25))
        correlated_values.append(new_value)
    
    correlated_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': correlated_values,
        'id': ids
    })
    
    # Create a non-correlated dataset
    # We'll create a completely different pattern
    non_correlated_values = []
    for timestamp in timestamps:
        # Generate random values between 50 and 300
        value = random.randint(50, 300)
        non_correlated_values.append(value)
    
    non_correlated_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': non_correlated_values,
        'id': ids
    })
    
    # Calculate correlation coefficients
    corr_coef = np.corrcoef(main_df['value'], correlated_df['value'])[0, 1]
    non_corr_coef = np.corrcoef(main_df['value'], non_correlated_df['value'])[0, 1]
    
    print(f"Generated {len(main_df)} rows of data")
    print(f"Correlation with correlated dataset: {corr_coef:.4f}")
    print(f"Correlation with non-correlated dataset: {non_corr_coef:.4f}")
    
    # Return the three datasets
    return main_df, correlated_df, non_correlated_df

def save_datasets(main_df, correlated_df, non_correlated_df, base_filename="dataset"):
    """Save the three datasets to CSV files"""
    main_df.to_csv(f"{base_filename}_main.csv", index=False)
    correlated_df.to_csv(f"{base_filename}_correlated.csv", index=False)
    non_correlated_df.to_csv(f"{base_filename}_non_correlated.csv", index=False)
    
    print(f"Datasets saved to {base_filename}_main.csv, {base_filename}_correlated.csv, and {base_filename}_non_correlated.csv")

def main():
    # Generate datasets with at least 15,000 rows
    main_df, correlated_df, non_correlated_df = generate_datasets(15000)
    
    # Save the datasets
    save_datasets(main_df, correlated_df, non_correlated_df)
    
    # Show a sample of each dataset
    print("\nMain dataset sample:")
    print(main_df.head())
    
    print("\nCorrelated dataset sample:")
    print(correlated_df.head())
    
    print("\nNon-correlated dataset sample:")
    print(non_correlated_df.head())
    
    # Calculate and print additional statistics
    print("\nMain dataset statistics:")
    print(main_df['value'].describe())
    
    print("\nCorrelated dataset statistics:")
    print(correlated_df['value'].describe())
    
    print("\nNon-correlated dataset statistics:")
    print(non_correlated_df['value'].describe())

if __name__ == "__main__":
    main()