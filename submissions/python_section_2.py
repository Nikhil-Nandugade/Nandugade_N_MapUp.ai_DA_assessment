#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Question 9: Distance Matrix Calculation

# In[2]:


df = pd.read_csv(r"D:\Imarticus\Nandugade_Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv")

def calculate_distance_matrix(df):
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  
        
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

# Distance matrix
distance_matrix = calculate_distance_matrix(df)
distance_matrix


# # Question 10: Unroll Distance Matrix

# In[14]:


def unroll_distance_matrix(distance_matrix):
    unrolled_data = []
    
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            if i != j:  
                unrolled_data.append({'id_start': i, 'id_end': j, 'distance': distance_matrix.at[i, j]})
    
    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df

unrolled_distance_df = unroll_distance_matrix(distance_matrix)
unrolled_distance_df


# # Question 11: Finding IDs within Percentage Threshold

# In[13]:


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    reference_distances = unrolled_df[unrolled_df['id_start'] == reference_id]['distance']
    
    if reference_distances.empty:
        return [] 
    average_distance = reference_distances.mean()
    
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    ids_within_threshold = unrolled_df[(unrolled_df['distance'] >= lower_bound) & 
                                       (unrolled_df['distance'] <= upper_bound)]
    
    return sorted(ids_within_threshold['id_start'].unique())

# Example 
# reference_id_example = 1001400
# ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_matrix, reference_id_example)
# ids_within_threshold


# # Question 12: Calculate Toll Rate

# In[15]:


def calculate_toll_rate(unrolled_df):
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, rate in rates.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * rate
    
    return unrolled_df

# Example 
#toll_rate_df = calculate_toll_rate(unrolled_distance_matrix)
#toll_rate_df


# # Question 13: Calculate Time-Based Toll Rates

# In[17]:


import datetime

def calculate_time_based_toll_rates(toll_rate_df):
    expanded_rows = []
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    time_ranges = [
        (datetime.time(0, 0), datetime.time(10, 0), 0.8),   
        (datetime.time(10, 0), datetime.time(18, 0), 1.2),  
        (datetime.time(18, 0), datetime.time(23, 59, 59), 0.8),  
    ]
    weekend_discount = 0.7
    for (id_start, id_end), group in toll_rate_df.groupby(['id_start', 'id_end']):
        distance = group['distance'].iloc[0]  
        for day in days_of_week:
            if day in days_of_week[:5]:  
                for start_time, end_time, discount in time_ranges:
                    new_row = {
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': 'Friday' if day == 'Monday' else days_of_week[days_of_week.index(day)+1],  
                        'end_time': end_time,
                    }
                   
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        new_row[vehicle] = group[vehicle].iloc[0] * discount
                    expanded_rows.append(new_row)
            else:  
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': datetime.time(0, 0),
                    'end_day': day,
                    'end_time': datetime.time(23, 59, 59),
                }
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    new_row[vehicle] = group[vehicle].iloc[0] * weekend_discount
                expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    
    return expanded_df

# Example 
#time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rate_df)
#time_based_toll_rates_df

