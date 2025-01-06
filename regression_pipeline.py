from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import shap
import joblib
import sys
import os

def evaluate_model_with_k_features(k, indices, X_train, y_train, pipeline):
    loo = LeaveOneOut()
    X_train_k = X_train.iloc[:, indices[:k]]
    scores = cross_val_score(pipeline, X_train_k, y_train, cv=loo, scoring='neg_mean_squared_error')
    return np.mean(scores)

def regression_process_fold(random_seed, fold, X_train, X_test, y_train, y_test, metric, learning_task, task_output, target_col=None):
    print(f"Processing fold {fold}{' for target ' + str(target_col) if target_col is not None else ''}")

    # find best subset of features
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  
        ('regressor', RandomForestRegressor(random_state=random_seed))  
    ])
    # pipeline.fit(X_train, y_train)
    # feature_importances = pipeline.named_steps['regressor'].feature_importances_
    # sorted_feature_importance_indices = np.argsort(feature_importances)[::-1]
    # print(sorted_feature_importance_indices)
    # results = []
    # # for k in range(1, X_train.shape[1] + 1):
    # for k in range(5, 30):
    #     score = evaluate_model_with_k_features(k, sorted_feature_importance_indices, X_train, y_train, pipeline)
    #     print("k = ", k, " score = ", score)
    #     results.append((k, score))
    # results_df = pd.DataFrame(results, columns=['K', 'Score'])
    # best_k = results_df.loc[results_df['Score'].idxmax()][0]
    # best_k = int(best_k)
   

    # # train best model with K features
    # X_train_k = X_train.iloc[:, sorted_feature_importance_indices[:best_k]]
    # X_test_k = X_test.iloc[:, sorted_feature_importance_indices[:best_k]]

    X_train_k = X_train
    X_test_k = X_test

    pipeline.fit(X_train_k, y_train)

    pipeline_dummy = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  
        ('regressor', DummyRegressor(strategy="mean"))
    ])

    dummy_regressor = pipeline_dummy.fit(X_train_k, y_train)
    
    # Make predictions
    y_pred_train, y_pred_dummy_train = pipeline.predict(X_train_k), dummy_regressor.predict(X_train_k)
    y_pred_test, y_pred_dummy_test = pipeline.predict(X_test_k), dummy_regressor.predict(X_test_k)

    if task_output == 'multi':
        # Calculate MSE for each target in the multi-output case
        train_mse = mean_squared_error(y_train, y_pred_train, multioutput='raw_values')
        test_mse = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
        train_mse_dummy = mean_squared_error(y_train, y_pred_dummy_train, multioutput='raw_values')
        test_mse_dummy = mean_squared_error(y_test, y_pred_dummy_test, multioutput='raw_values')
    else:
        # Calculate a single MSE for single-output tasks
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mse_dummy = mean_squared_error(y_train, y_pred_dummy_train)
        test_mse_dummy = mean_squared_error(y_test, y_pred_dummy_test)
    
    # Initialize SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(pipeline.named_steps['regressor'], X_train_k)
    shap_values_train = explainer.shap_values(X_train_k, check_additivity=False)
    shap_values_test = explainer.shap_values(X_test_k, check_additivity=False)
    
    # File naming
    suffix = f"_{target_col}" if target_col is not None else ""
  
    # Save model, predictions, and SHAP values
    joblib.dump(pipeline, os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/models', f"model_fold_{fold}{suffix}.joblib"))
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/predictions',  f"train_predictions_fold_{fold}{suffix}.npy"), y_pred_train) 
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/predictions',  f"test_predictions_fold_{fold}{suffix}.npy"), y_pred_test) 
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/shap', f"train_shap_fold_{fold}{suffix}.npy"), shap_values_train)  
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/shap', f"test_shap_fold_{fold}{suffix}.npy"), shap_values_test)  
    
    return {
        'fold': fold,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mse_dummy': train_mse_dummy,
        'test_mse_dummy': test_mse_dummy,
        # 'features_used': sorted_feature_importance_indices[:best_k]
    }


def regression_training_pipeline(metric, learning_task, task_output, random_seed):
    # load data
    df_x = pd.read_csv('./processed_data/metafeatures.csv', index_col=0)
    df_y = pd.read_csv(f'./processed_data/{metric}/{learning_task}/performance.csv', index_col=0)

    # Create directories for saving results
    directories = ['models', 'shap', 'predictions']
    for dir in directories:
        os.makedirs(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/{dir}/', exist_ok=True)

    loo = LeaveOneOut()
    results = []
    if(task_output=='multi'):
        for fold, (train_index, test_index) in enumerate(loo.split(df_x)):
            X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
            y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
            result = regression_process_fold(random_seed, fold, X_train, X_test, y_train, y_test, metric, learning_task, task_output)
            results.append(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./results/seed_{random_seed}/{metric}/{learning_task}/{task_output}/performance_results.csv', index=False)
    elif(task_output=='single'):
       with ProcessPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(train_and_evaluate_target, random_seed, df_x, df_y, metric, learning_task, task_output, col): col for col in df_y.columns}
            for future in as_completed(futures):
                target_col, target_results = future.result()
                results_df = pd.DataFrame(target_results)
                results_df.to_csv(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/performance_results_{target_col}.csv', index=False)

def train_and_evaluate_target(random_seed, df_x, df_y, metric, learning_task, task_output, target_col):
    target_results = []
    loo = LeaveOneOut()
    for fold, (train_index, test_index) in enumerate(loo.split(df_x)):
        X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
        y_train, y_test = df_y.iloc[train_index][target_col], df_y.iloc[test_index][target_col]
        target_result = regression_process_fold(random_seed, fold, X_train, X_test, y_train, y_test, metric, learning_task, task_output, target_col)
        target_results.append(target_result)
    return target_col, target_results

if __name__ == "__main__":
    # for random_seed in range(0, 10):
    random_seed = int(sys.argv[1])
    metric = sys.argv[2]
    learning_task = sys.argv[3]
    task_output = sys.argv[4]
    regression_training_pipeline(metric, learning_task, task_output, random_seed)