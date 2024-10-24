#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Question 1: Reverse List by N Elements

# In[2]:


def reverse_by_n_elements(lst, n):
    result = []
    for i in range(0, len(lst), n):
        temp = []
        for j in range(min(n, len(lst) - i)):
            temp.insert(0, lst[i + j])
        result.extend(temp)
    return result

# Example
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))   
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))            
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  


# # Question 2: Lists & Dictionaries

# In[3]:


lst1 = ["apple", "bat", "car", "elephant", "dog", "bear"]

def len_word(word_lst):
    keys = list(set([len(i) for i in word_lst]))
    keys.sort() 
    result = dict()
    for i in keys:
        result[i] = [j for j in word_lst if len(j) == i]
    return print(result)

len_word(lst1)


# # Question 3: Flatten a Nested Dictionary

# In[4]:


def flatten_dict(d, parent_key='', sep='.'):
    flattened = {}
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                flattened.update(flatten_dict({f"{new_key}[{i}]": item}, '', sep=sep))
        else:
            flattened[new_key] = v
    
    return flattened

# example
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}


flattened_dict = flatten_dict(nested_dict)
flattened_dict


# # Question 4: Generate Unique Permutations

# In[5]:


def backtrack(nums, path, used, res):
    if len(path) == len(nums):
        res.append(path[:])
        return
    
    for i in range(len(nums)):
        if used[i]:
            continue
        if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
            continue
        used[i] = True
        path.append(nums[i])
        backtrack(nums, path, used, res)
        path.pop()
        used[i] = False

def unique_permutations(nums):
    nums.sort()  
    res = []
    used = [False] * len(nums)
    backtrack(nums, [], used, res)
    return res

# Example
nums = [1, 1, 2]
unique_perms = unique_permutations(nums)
print(unique_perms)


# # Question 5: Find All Dates in a Text

# In[6]:


import re

def find_all_dates(text):
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    dates = re.findall(date_pattern, text)
    
    return dates

# Example
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
found_dates = find_all_dates(text)
print(found_dates)


# # Question 6: Decode Polyline, Convert to DataFrame with Distances

# In[7]:


import polyline

def haversine(coord1, coord2):
    R = 6371000 
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def decode_polyline_and_create_dataframe(polyline_str):
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    distances = [0]  
    for i in range(1, len(coordinates)):
        distance = haversine(coordinates[i-1], coordinates[i])
        distances.append(distance)
    
    df['distance'] = distances
    return df

# example
polyline_str = "a~l~F~z`~w@q@kB"
df = decode_polyline_and_create_dataframe(polyline_str)
print(df)


# # Question 7: Matrix Rotation and Transformation

# In[8]:


def rotate_and_transform(matrix):
    n = len(matrix)
    rotated = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]

    transformed = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            transformed[i][j] = row_sum + col_sum

    return transformed

# Example
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

final_matrix = rotate_and_transform(matrix)
for row in final_matrix:
    print(row)


# # Question 8: Time Check

# In[9]:


import datetime

weekday_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

def get_datetime(day, time, reference_date):
    weekday_num = weekday_mapping[day]
    days_diff = weekday_num - reference_date.weekday()
    date = reference_date + datetime.timedelta(days=days_diff)
    time_part = datetime.datetime.strptime(time, '%H:%M:%S').time()
    return datetime.datetime.combine(date, time_part)

def check_time_completeness(df):
    reference_date = datetime.datetime(2024, 10, 21)  
    df['start_datetime'] = df.apply(lambda row: get_datetime(row['startDay'], row['startTime'], reference_date), axis=1)
    df['end_datetime'] = df.apply(lambda row: get_datetime(row['endDay'], row['endTime'], reference_date), axis=1)
    
    grouped = df.groupby(['id', 'id_2'])
    
    completeness_series = pd.Series(dtype=bool, index=pd.MultiIndex.from_tuples([], names=["id", "id_2"]))
    
    for (id_val, id_2_val), group in grouped:
        days_covered = group['startDay'].map(weekday_mapping).unique()
        all_days_covered = set(days_covered) == set(range(7)) 
        
        full_day_coverage = True
        for day in set(days_covered):
            day_group = group[(group['startDay'].map(weekday_mapping) == day)]
            day_start_min = day_group['start_datetime'].min().time()
            day_end_max = day_group['end_datetime'].max().time()
            if not (day_start_min == datetime.time(0, 0, 0) and day_end_max == datetime.time(23, 59, 59)):
                full_day_coverage = False
                break
        completeness_series.loc[(id_val, id_2_val)] = not (all_days_covered and full_day_coverage)
    
    return completeness_series

# Example 
df = pd.read_csv(r"D:\Imarticus\Nandugade_Mapup\MapUp-DA-Assessment-2024\datasets\dataset-1.csv")
result = check_time_completeness(df)
print(result)

