# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:11:07 2024

@author: jettr
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#%%
# Load the dataset, specifying low_memory=False to address the mixed types warning
## Make sure second row is deleted.
file_path = 'C:/Users/jettr/Dropbox (University of Oregon)/23-24/Winter/Erth 451 Hydro/Hydro FInal Project/GWB mineral data/NWIS_Aq_wq_RugebregtOutput_MinSat_N400SLRDVN.csv'
data = pd.read_csv(file_path, low_memory=False)

#%%

# Fill missing Aquifer Code values with 'Unknown'
data['Aquifer Code'] = data['Aquifer Code'].fillna('Unknown')

# Convert columns to numeric and remove outliers
data[['pH', 'Temperature', 'Well Depth']] = data[['pH', 'Temperature', 'Well Depth']].apply(pd.to_numeric, errors='coerce')
data = data[(data['pH'] >= 0) & (data['pH'] <= 14) & (data['Temperature'] <= 100) & (data['Well Depth'] <= 10000)]
mineral_columns = data.columns[data.columns.get_loc('Akermanite'):]

# Convert the data of mineral columns to numeric
data[mineral_columns] = data[mineral_columns].apply(pd.to_numeric, errors='coerce')

# Calculate mean and median differences from 0 for minerals
mean_diffs = data[mineral_columns].abs().mean()
median_diffs = data[mineral_columns].abs().median()

# Identify the number of minerals closest to 0 by mean and median
mineral_num = 10
closest_means_minerals = mean_diffs.nsmallest(mineral_num).index.tolist()
closest_medians_minerals = median_diffs.nsmallest(mineral_num).index.tolist()

minerals_to_plot = list(set(closest_means_minerals) | set(closest_medians_minerals))

sns.set(style="whitegrid")

# Generate colors for each Aquifer Code
unique_aquifer_codes = sorted(set(data['Aquifer Code']), key=lambda x: ('Unknown' == x, x))
palette = sns.color_palette('husl', n_colors=len(unique_aquifer_codes) - 1) + ['gray']
aquifer_color_map = {code: color for code, color in zip(unique_aquifer_codes, palette)}

for mineral in minerals_to_plot:
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(left=0.2, right=0.85)  # Adjust subplot parameters to give space for legend

    for i, param in enumerate(['pH', 'Temperature', 'Well Depth']):
        subset = data.dropna(subset=[mineral, param])
        sns.regplot(data=subset, x=param, y=mineral, ax=axs[i], scatter=False, color="black", 
                    # ci=None
                    )

        for code, count in subset['Aquifer Code'].value_counts().items():
            subsubset = subset[subset['Aquifer Code'] == code]
            axs[i].scatter(subsubset[param], subsubset[mineral], color=aquifer_color_map[code], label=f"{code} (n={count})", alpha=0.7)

        correlation = subset[[param, mineral]].corr().iloc[0, 1]
        param_stats = subset[param].agg(['mean', 'median', 'min', 'max'])

        stats_text = (f'Corr: {correlation:.2f}\n'
                      f'{param} Mean: {param_stats["mean"]:.2f}, Median: {param_stats["median"]:.2f}, '
                      f'Min: {param_stats["min"]:.2f}, Max: {param_stats["max"]:.2f}\n'
                      f'Total Points: {len(subset)}')

        axs[i].annotate(stats_text, xy=(0.5, 1.15), xycoords='axes fraction', ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round", alpha=0.5, color='wheat'))
        axs[i].set_title(f'{param} vs. {mineral} Saturation')

    # Create a unified legend for the first subplot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Aquifer Code', loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=9)

    plt.show()
    
#%%

# Fill missing Aquifer Code values with 'Unknown'
data['Aquifer Code'] = data['Aquifer Code'].fillna('Unknown')

# Convert 'NO3-' column to numeric and remove rows where 'NO3-' is not greater than 0
data['NO3-'] = pd.to_numeric(data['NO3-'], errors='coerce')
data = data[data['NO3-'] > 0]

# Select mineral columns
mineral_columns = data.columns[data.columns.get_loc('Akermanite'):]

# Convert the data of mineral columns to numeric
data[mineral_columns] = data[mineral_columns].apply(pd.to_numeric, errors='coerce')

# Calculate absolute mean and median differences from 0 for minerals
mean_diffs = data[mineral_columns].abs().mean()
median_diffs = data[mineral_columns].abs().median()

# Identify the number of minerals closest to 0 by mean and median
mineral_num = 5
closest_means_minerals = mean_diffs.nsmallest(mineral_num).index.tolist()
closest_medians_minerals = median_diffs.nsmallest(mineral_num).index.tolist()

minerals_to_plot = list(set(closest_means_minerals) | set(closest_medians_minerals))

sns.set(style="whitegrid")

# Generate colors for each Aquifer Code
unique_aquifer_codes = sorted(set(data['Aquifer Code']), key=lambda x: ('Unknown' == x, x))
palette = sns.color_palette('husl', n_colors=len(unique_aquifer_codes) - 1) + ['gray']
aquifer_color_map = {code: color for code, color in zip(unique_aquifer_codes, palette)}

# Plotting loop for 'NO3-' parameter only
param = 'NO3-'
for mineral in minerals_to_plot:
    subset = data.dropna(subset=[mineral, param])
    plt.figure(figsize=(10, 6))

    sns.regplot(data=subset, x=param, y=mineral, scatter=False, color="black")
    
    for code, count in subset['Aquifer Code'].value_counts().items():
        subsubset = subset[subset['Aquifer Code'] == code]
        plt.scatter(subsubset[param], subsubset[mineral], color=aquifer_color_map[code], label=f"{code} (n={count})", alpha=0.7)

        correlation = subsubset[[param, mineral]].corr().iloc[0, 1]
        param_stats = subsubset[param].agg(['mean', 'median', 'min', 'max'])

        stats_text = (f'Corr: {correlation:.2f}\n'
                      f'{param} Mean: {param_stats["mean"]:.2f}, Median: {param_stats["median"]:.2f}, '
                      f'Min: {param_stats["min"]:.2f}, Max: {param_stats["max"]:.2f}\n'
                      f'Total Points: {len(subsubset)}')

        plt.annotate(stats_text, xy=(0.5, 1.05), xycoords='axes fraction', ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round", alpha=0.5, color='wheat'))

    plt.title(f'{param} vs. {mineral} Saturation')
    plt.xlabel(param)
    plt.ylabel(mineral)
    plt.legend(title='Aquifer Code')
    plt.show()


#%%

file_path2 = 'C:/Users/jettr/Dropbox (University of Oregon)/23-24/Winter/Erth 451 Hydro/Hydro FInal Project/GWB mineral data/Minerals_wExtra.csv'
data2 = pd.read_csv(file_path2, low_memory=False)

# Fill missing Aquifer Code values with 'Unknown'
data2['Aquifer Code'] = data2['Aquifer Code'].fillna('Unknown')

# Convert 'Arsenic' column to numeric and remove rows where 'NO3-' is not greater than 0
data2['Arsenic'] = pd.to_numeric(data2['Arsenic'], errors='coerce')
data2 = data2[data2['NO3-'] > 0]

# Select mineral columns
mineral_columns = data2.columns[data2.columns.get_loc('Akermanite'):]

# Convert the data of mineral columns to numeric
data2[mineral_columns] = data2[mineral_columns].apply(pd.to_numeric, errors='coerce')

# Calculate absolute mean and median differences from 0 for minerals
mean_diffs = data2[mineral_columns].abs().mean()
median_diffs = data2[mineral_columns].abs().median()

# Identify the number of minerals closest to 0 by mean and median
mineral_num = 5
closest_means_minerals = mean_diffs.nsmallest(mineral_num).index.tolist()
closest_medians_minerals = median_diffs.nsmallest(mineral_num).index.tolist()

minerals_to_plot = list(set(closest_means_minerals) | set(closest_medians_minerals))

sns.set(style="whitegrid")

# Generate colors for each Aquifer Code
unique_aquifer_codes = sorted(set(data2['Aquifer Code']), key=lambda x: ('Unknown' == x, x))
palette = sns.color_palette('husl', n_colors=len(unique_aquifer_codes) - 1) + ['gray']
aquifer_color_map = {code: color for code, color in zip(unique_aquifer_codes, palette)}

# Plotting loop for 'NO3-' parameter only
param = 'Arsenic'
for mineral in minerals_to_plot:
    subset = data2.dropna(subset=[mineral, param])
    plt.figure(figsize=(10, 6))

    sns.regplot(data=subset, x=param, y=mineral, scatter=False, color="black")
    
    for code, count in subset['Aquifer Code'].value_counts().items():
        subsubset = subset[subset['Aquifer Code'] == code]
        plt.scatter(subsubset[param], subsubset[mineral], color=aquifer_color_map[code], label=f"{code} (n={count})", alpha=0.7)

        correlation = subsubset[[param, mineral]].corr().iloc[0, 1]
        param_stats = subsubset[param].agg(['mean', 'median', 'min', 'max'])

        stats_text = (f'Corr: {correlation:.2f}\n'
                      f'{param} Mean: {param_stats["mean"]:.2f}, Median: {param_stats["median"]:.2f}, '
                      f'Min: {param_stats["min"]:.2f}, Max: {param_stats["max"]:.2f}\n'
                      f'Total Points: {len(subsubset)}')

        plt.annotate(stats_text, xy=(0.5, 1.05), xycoords='axes fraction', ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round", alpha=0.5, color='wheat'))

    plt.title(f'{param} vs. {mineral} Saturation')
    plt.xlabel(param)
    plt.ylabel(mineral)
    plt.legend(title='Aquifer Code' , bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
