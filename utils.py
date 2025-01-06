import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def get_algo_portfolio(metric, top_k, datasets, find_min):
    df = pd.read_csv("./data/performance.csv", index_col=0)
    # Create a filtered copy of the DataFrame with only the necessary columns
    df_filtered = df[['DATASET', 'METHOD', metric]].copy()

    df_filtered['DATASET'] = df_filtered['DATASET'].str.upper() 
    df_filtered = df_filtered[df_filtered['DATASET'].isin(datasets)]

    # Determine if each row is the best within its group based on find_min
    if find_min:
        df_filtered['Is_Best'] = df_filtered.groupby('DATASET')[metric].transform(lambda x: x == x.min())
    else:
        df_filtered['Is_Best'] = df_filtered.groupby('DATASET')[metric].transform(lambda x: x == x.max())

    # Count how many times each method was the best and reset the index to turn it into a DataFrame
    best_counts = df_filtered[df_filtered['Is_Best'] == True]['METHOD'].value_counts().reset_index()
    best_counts.columns = ['METHOD', 'Best_Count']

    # Get the top k algorithms based on the count of being the best
    top_k_algos = best_counts.head(top_k)['METHOD'].values

    # Convert the best_counts DataFrame into a dictionary for easier access
    counts_dict = pd.Series(best_counts.Best_Count.values,index=best_counts.METHOD).to_dict()

    # Return the array of top k algorithms and the dictionary with their counts
    # print("Top k algorithms:", algo_portfolio)
    # print("Algorithm counts:", algo_counts)
    return top_k_algos, counts_dict
    

def prepare_metafeatures_and_regression_performance_data(metric, find_min, top_k):
    # load and prepare data
    df_x = pd.read_csv('./data/metafeatures.csv', index_col=0)
    df_y = pd.read_csv("./data/performance.csv", index_col=0)
    datasets = list(df_x.index.str.upper()) 
    df_y['DATASET'] = df_y['DATASET'].str.upper() 
    df_y = df_y[df_y['DATASET'].isin(datasets)]
    df_y = df_y[['DATASET', 'METHOD', metric]]
    df_y = df_y.pivot(index='DATASET', columns='METHOD', values=metric)

    algo_portfolio, algo_counts = get_algo_portfolio(metric, top_k, datasets, find_min)
    df_y = df_y[algo_portfolio]

    # impute missing values
    columns_with_missing_values = df_y.columns[df_y.isnull().any()]
    for column_with_missing_value in columns_with_missing_values:
        mean_value = df_y[column_with_missing_value].mean()
        df_y[column_with_missing_value].fillna(value=mean_value, inplace=True)
    
    return df_x, df_y, algo_portfolio, algo_counts



def calculate_gap_closed(metric, results_dir, seed):
    # print(metric)
    algos = np.load(f'./processed_data/{metric}/algo_portfolio.npy', allow_pickle=True)
    df = pd.read_csv(f'./{results_dir}/seed_{seed}/{metric}/AS.csv', index_col=0)

    approaches = ['AS-R-MO', 'AS-R-SO',  'AS-PR-MO', 'AS-PR-SO', 'AS-C-SO', 'AS-PC-MO', 'AS-PC-SO', 'AS-CS-PC-SO']

    # Dictionary to store the total loss of each SBS algorithm
    total_losses = {}
    
    # Calculate the total loss for each algorithm
    for algo in algos:
        total_loss = np.sum(df['SBS_'+algo])
        total_losses[algo] = total_loss
        # print(f'SBS_{algo}: {total_loss}')
    
    # Find the algorithm with the smallest total loss
    sbs_min_loss_algo = min(total_losses, key=total_losses.get)
    min_loss = total_losses[sbs_min_loss_algo]
    
    data_gap = []
    for approach in approaches:
        total_loss_approach =  np.sum(df[approach])
        gap = (min_loss-total_loss_approach)/min_loss*100
        data_gap.append(gap)
        # print(approach, (min_loss-total_loss_approach)/min_loss)
    return data_gap

def calculate_gap_closed_with_repetitions( metric):
    approaches = ['AS-R-MO', 'AS-R-SO', 'AS-PR-MO', 'AS-PR-SO', 'AS-C-SO', 'AS-PC-MO', 'AS-PC-SO', 'AS-CS-PC-SO']
    data_gaps = {approach: [] for approach in approaches}  # Initialize a dict to hold gaps for each approach

    for seed in range(0, 1):
        algos = np.load(f'./processed_data/{metric}/algo_portfolio.npy', allow_pickle=True)
        df = pd.read_csv(f'./results/seed_{seed}/{metric}/AS.csv', index_col=0)
        # print(df)

        # Dictionary to store the total loss of each SBS algorithm
        total_losses = {}
        
        # Calculate the total loss for each algorithm
        for algo in algos:
            total_loss = np.mean(df['SBS_'+algo])
            total_losses[algo] = total_loss
        
        # Find the algorithm with the smallest total loss
        sbs_min_loss_algo = min(total_losses, key=total_losses.get)
        min_loss = total_losses[sbs_min_loss_algo]
        
        # Calculate the gap for each approach and add it to the respective list
        for approach in approaches:
            total_loss_approach = np.mean(df[approach])
            gap = (min_loss - total_loss_approach) / min_loss
            data_gaps[approach].append(gap)
    
    # Calculate mean and standard deviation for each approach across seeds
    results = {approach: {'mean': np.mean(gaps), 'std': np.std(gaps)} for approach, gaps in data_gaps.items()}
    
    return results