import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from scipy.stats import randint
import pickle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from utils import split_dataset

def plot_roc_auc_curve_ovr(models, y_score, y_train, y_test):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    n_classes = len(pd.unique(y_test))
    n_models = len(models)
    for i in range(n_classes):
        fig, ax = plt.subplots(figsize=(6, 6))
        class_id = np.flatnonzero(label_binarizer.classes_ == i)[0]
        for m in range(n_models):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_score[m][:, class_id],
                name=models[m],
                ax=ax
            )

        plt.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate", fontweight='bold')
        plt.ylabel("True Positive Rate", fontweight='bold')
        plt.title(f"Class {i} vs Rest", fontweight='bold')
        plt.legend()
        plt.savefig(f"./final_plots/roc_curve_{i}.eps", dpi=400, bbox_inches='tight')
        plt.show()


max_feature_1 = 'sqrt'
max_feature_2 = 'log2'

# Define the parameter grid for each model
param_grid_rf = {
    "n_estimators": randint(60, 1000),
    "max_depth": randint(3, 21),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": [max_feature_1, max_feature_2, None],
    "bootstrap": [True, False],
    "random_state": [0]
}

param_grid_gbc = {
    "n_estimators": randint(60, 1000),
    "max_depth": randint(3, 21),
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": [max_feature_1, max_feature_2, None],
    "random_state": [0]
}

xgb_grid = {
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": range(3, 21, 2),
    # Gamma specifies the minimum loss reduction required to make a split.
    "gamma": [i / 10.0 for i in range(0, 5)],
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [i / 10.0 for i in range(3, 10)],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100], 'n_estimators': range(60, 1000, 50), }


traditional_models = ['Gradient Boosting', 'Random Forest', 'XGBoost']
# Set up the models
rf_model = RandomForestClassifier()
gbc_model = GradientBoostingClassifier()
xgb_model = xgb.XGBClassifier()

# Set up the parameter grid for grid search
param_grid = {
    'Random Forest': param_grid_rf,
    'Gradient Boosting': param_grid_gbc,
    'XGBoost': xgb_grid,
}

mooc_df = pd.read_csv("./data/grade_prediction_mooc.csv")

mooc_with_node2vec = pd.read_csv('./data/mooc_with_node2vec_embedding.csv')
mooc_with_deepwalk = pd.read_csv('./data/mooc_with_deepwalk_embedding.csv')

all_test_probs = []
model_titles = []
for dataset_version in ['base', 'node2vec', 'Deepwalk']:

    X_train, X_test, y_train, y_test = split_dataset(
        mooc_df if dataset_version == 'base' else mooc_with_node2vec if dataset_version == 'node2vec' else mooc_with_deepwalk)
    for model_name, model in zip(traditional_models,
                                 [gbc_model, rf_model, xgb_model]):

        model.fit(X_train, y_train)

        pickle.dump(model, open(f'./models/{dataset_version}_{model_name}_classifier.sav', 'wb'))
        model_title = model_name if dataset_version == 'base' else model_name + ' + ' + dataset_version
        model_titles.append(model_title)

        print(f"Results for {model_title}:")
        print(f"Model params: {model.get_params()}")

        test_probs = model.predict_proba(X_test)
        all_test_probs.append(test_probs)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))


plot_roc_auc_curve_ovr(model_titles, all_test_probs, y_train, y_test)
