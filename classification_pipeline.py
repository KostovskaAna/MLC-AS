import os
import numpy as np
import pandas as pd
import shap
import sys
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier, clone
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# for target_col in [col for col in df_y.columns if "_cost" not in col]:
#     results = []
#     for fold, (train_index, test_index) in enumerate(loo.split(df_x)):
#         X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
#         y_train, y_test = df_y.iloc[train_index][target_col], df_y.iloc[test_index][target_col]
#         y_train_cost = df_y.iloc[train_index][target_col + "_cost"] if cost_sensitive else None
#         result = classification_process_fold(fold, X_train, X_test, y_train, y_test, pipeline, metric, learning_task, task_output, target_col, cost_sensitive, y_train_cost)
#         results.append(result)
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(f'./results_reduced_top5_5/{metric}/{learning_task}/{task_output}/performance_results_{target_col}.csv', index=False)
def multioutput_accuracy(y_true, y_pred):
    """Calculate average accuracy across all outputs."""
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate accuracy for each output
    accuracies = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return np.mean(accuracies)

def cross_val_score_multioutput(model, X, y, scoring=None):
    """A custom cross_val_score function for multi-output data."""
    kf = LeaveOneOut()
    scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        y_pred = cloned_model.predict(X_test)
        
        score = scoring(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)

class WeightedRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)

def create_pipeline(cost_sensitive, random_seed):
    """Create and return a pipeline based on cost sensitivity."""
    classifier = WeightedRandomForestClassifier if cost_sensitive else RandomForestClassifier
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),   
        ('classifier', classifier(random_state=random_seed))
    ])
def evaluate_model_with_k_features(task_output, k, indices, X_train, y_train, pipeline):
    loo = LeaveOneOut()
    X_train_k = X_train.iloc[:, indices[:k]]
    if task_output == 'multi':
        scores = cross_val_score_multioutput(pipeline, X_train_k, y_train, scoring=multioutput_accuracy)
    else:
        scores = cross_val_score(pipeline, X_train_k, y_train, cv=loo, scoring='accuracy')
    return np.mean(scores)


def classification_process_fold(random_seed, fold, X_train, X_test, y_train, y_test, metric, learning_task, task_output, target_col=None, cost_sensitive=False, y_train_cost=None):
    print(f"Processing fold {fold}{' for target ' + str(target_col) if target_col is not None else ''}")
    

    pipeline = create_pipeline(cost_sensitive, random_seed)
    if cost_sensitive:
        pipeline.fit(X_train, y_train, classifier__sample_weight=y_train_cost)
    elif learning_task == 'classification':
        best_algo = np.load(f'./processed_data/{metric}/algo_portfolio.npy', allow_pickle=True)[0]
        pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)
    # # # find best subset of features
    # feature_importances = pipeline.named_steps['classifier'].feature_importances_
    # sorted_feature_importance_indices = np.argsort(feature_importances)[::-1]
    # print(sorted_feature_importance_indices)
    # results = []
    # # for k in range(1, X_train.shape[1] + 1):
    # for k in range(5, 30):
    #     score = evaluate_model_with_k_features(task_output, k, sorted_feature_importance_indices, X_train, y_train, pipeline)
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


    # Fit models
    if cost_sensitive:
        pipeline.fit(X_train_k, y_train, classifier__sample_weight=y_train_cost)
        dummy_classifier = DummyClassifier(strategy="most_frequent").fit(X_train_k, y_train, sample_weight=y_train_cost)
    elif learning_task == 'classification':
        best_algo = np.load(f'./processed_data/{metric}/algo_portfolio.npy', allow_pickle=True)[0]
        pipeline.fit(X_train_k, y_train)
        dummy_classifier = DummyClassifier(strategy="constant", constant=best_algo).fit(X_train_k, y_train)
    else:
        pipeline.fit(X_train_k, y_train)
        dummy_classifier = DummyClassifier(strategy="most_frequent").fit(X_train_k, y_train)
    
    # Make predictions
    y_pred_train, y_pred_dummy_train = pipeline.predict(X_train_k), dummy_classifier.predict(X_train_k)
    y_pred_test, y_pred_dummy_test = pipeline.predict(X_test_k), dummy_classifier.predict(X_test_k)

   
    if task_output == 'multi':  
        y_pred_train = pd.DataFrame(y_pred_train, index=y_train.index, columns=y_train.columns)
        y_pred_dummy_train = pd.DataFrame(y_pred_dummy_train, index=y_train.index, columns=y_train.columns)
        y_pred_test = pd.DataFrame(y_pred_test, index=y_test.index, columns=y_test.columns)
        y_pred_dummy_test = pd.DataFrame(y_pred_dummy_test, index=y_test.index, columns=y_test.columns)
        train_accuracy = [accuracy_score(y_train[col], y_pred_train[col]) for col in y_train]
        train_accuracy_dummy = [accuracy_score(y_train[col], y_pred_dummy_train[col]) for col in y_train]
        test_accuracy = [accuracy_score(y_test[col], y_pred_test[col]) for col in y_test]
        test_accuracy_dummy = [accuracy_score(y_test[col], y_pred_dummy_test[col]) for col in y_test]
    else:
        y_train = list(y_train)
        y_test = list(y_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_accuracy_dummy = accuracy_score(y_train, y_pred_dummy_train)
        test_accuracy_dummy = accuracy_score(y_test, y_pred_dummy_test)

    print("acc: ", test_accuracy)
    # Initialize SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'], X_train)
    print("shap: ", X_train.shape, X_test.shape)
    shap_values_train = explainer.shap_values(np.array(X_train), check_additivity=False)
    shap_values_test = explainer.shap_values(np.array(X_test), check_additivity=False)

    # File naming
    suffix = f"_{target_col}" if target_col is not None else ""
  
    # Save model, predictions, and SHAP values
    joblib.dump(pipeline, os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/models', f"model_fold_{fold}{suffix}.joblib"))
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/predictions',  f"train_predictions_fold_{fold}{suffix}.npy"), y_pred_train) 
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/predictions',  f"test_predictions_fold_{fold}{suffix}.npy"), y_pred_test) 
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/shap', f"train_shap_fold_{fold}{suffix}.npy"), shap_values_train)  
    np.save(os.path.join(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/shap', f"test_shap_fold_{fold}{suffix}.npy"), shap_values_test)  
    
    return {
        'fold': fold,
        'train_score': train_accuracy,
        'test_score': test_accuracy,
        'train_score_dummy': train_accuracy_dummy,
        'test_score_dummy': test_accuracy_dummy,
        # 'features_used': sorted_feature_importance_indices[:best_k]
    }



def classification_training_pipeline(metric, learning_task, task_output, cost_sensitive,random_seed):
    print(task_output)
    # load data
    df_x = pd.read_csv('./processed_data/metafeatures.csv', index_col=0)
    df_y_file_path = f'./processed_data/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/performance.csv'
    df_y = pd.read_csv(df_y_file_path, index_col=0)

    # Create directories for saving results
    directories = ['models', 'shap', 'predictions']
    for dir in directories:
        directory_path = f'./results/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/{dir}/'
        os.makedirs(directory_path, exist_ok=True)
            


    loo = LeaveOneOut()
    
    results = []
    if learning_task == 'classification' or (learning_task == 'pairwise_classification' and task_output == 'multi'):
        for fold, (train_index, test_index) in enumerate(loo.split(df_x)):
            X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
            if(learning_task=='classification'):
                y_train, y_test = df_y.iloc[train_index, -1], df_y.iloc[test_index, -1]
            elif(learning_task == 'pairwise_classification' and task_output == 'multi'):
                y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
            result = classification_process_fold(random_seed, fold, X_train, X_test, y_train, y_test, metric, learning_task, task_output)
            results.append(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{learning_task}/{task_output}/performance_results.csv', index=False)
    elif learning_task == 'pairwise_classification' and task_output == 'single':
        with ProcessPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(train_and_evaluate_target, random_seed, df_x, df_y, metric, learning_task, task_output, cost_sensitive, col): col for col in df_y.columns  if "_cost" not in col}
            for future in as_completed(futures):
                target_col, target_results = future.result()
                results_df = pd.DataFrame(target_results)
                results_df.to_csv(f'./results_reduced_top5_5/seed_{random_seed}/{metric}/{"cost_sensitive_" if cost_sensitive else ""}{learning_task}/{task_output}/performance_results_{target_col}.csv', index=False)



def train_and_evaluate_target(random_seed, df_x, df_y, metric, learning_task, task_output, cost_sensitive, target_col):
    target_results = []
    loo = LeaveOneOut()
    for fold, (train_index, test_index) in enumerate(loo.split(df_x)):
        X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
        y_train, y_test = df_y.iloc[train_index][target_col], df_y.iloc[test_index][target_col]
        y_train_cost = df_y.iloc[train_index][target_col + "_cost"] if cost_sensitive else None
        result = classification_process_fold(random_seed, fold, X_train, X_test, y_train, y_test, metric, learning_task, task_output, target_col, cost_sensitive, y_train_cost)
        target_results.append(result)
    return target_col, target_results

if __name__ == "__main__":
    # for random_seed in range(0, 10):
        random_seed =  int(sys.argv[1])
        metric = sys.argv[2]
        learning_task = sys.argv[3]
        task_output = sys.argv[4]
        cost_sensitive = sys.argv[5].lower() 
        cost_sensitive = True if cost_sensitive == 'true' else False
        classification_training_pipeline(metric, learning_task, task_output, cost_sensitive, random_seed)
