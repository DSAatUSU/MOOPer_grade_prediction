import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.feature_selection import SelectFromModel
from utils import split_dataset

model_name = 'Gradient Boosting'

mooc_df = pd.read_csv("./data/grade_prediction_mooc.csv")

mooc_with_node2vec = pd.read_csv('./data/mooc_with_node2vec_embedding.csv')
mooc_with_deepwalk = pd.read_csv('./data/mooc_with_deepwalk_embedding.csv')


def compute_feature_importance(X_train, y_train, best_model, model_name):
    sel = SelectFromModel(best_model)

    sel.fit(X_train, y_train)
    sel.get_support()

    importances = sel.estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    importances[::-1].sort()

    feature_names = [list(X_train.columns)[i] for i in indices]

    if len(indices) >= 10:
        importance_list = []
        feature_names = []
        user_embedding_importance = 0
        challenge_embedding_importance = 0
        for idx, item in enumerate(indices):
            if item >= 138:
                challenge_embedding_importance = challenge_embedding_importance + importances[idx]
            elif item >= 10:
                user_embedding_importance = user_embedding_importance + importances[idx]
            else:
                feature_names.append(list(X_train.columns)[item])
                importance_list.append(importances[idx])
        # %%
        importance_list.insert(0, challenge_embedding_importance)
        importance_list.insert(1, user_embedding_importance)

        feature_names.insert(0, 'Challenge Embedding')
        feature_names.insert(1, 'User Embedding')
        np.argsort(importance_list)
        feature_names = [feature_names[i] for i in np.flip(np.argsort(importance_list))]
        importance_list.sort(reverse=True)
        # %%
        importances = np.array(importance_list)

    data = pd.DataFrame({"importance": importances.ravel(), 'names': feature_names})
    pal = sns.dark_palette("#69d", len(data), reverse=True)
    data['importance'] = data['importance'].astype(float)
    plot = sns.barplot(data=data, x="names", y='importance', palette=np.array(pal[::-1]))
    for item in plot.get_xticklabels():
        item.set_rotation(90)
        item.set_fontweight('bold')
    plt.xlabel('')
    plt.ylabel('Feature Importance', fontweight='bold')
    plt.title(f'Feature importance for {model_name}')
    plt.tight_layout()

    plt.show()


def showcase_user_categories(X_train_users, y_train, all_results):
    X_whole = pd.concat([X_train_users, y_train], axis=1)
    g = X_whole.groupby('user_id')
    test_ranges = [('Extremely\nLow', 0.9, 1), ('Very Low', 0.8, 0.9), ('Low', 0.5, 0.8), ('Average', 0.2, 0.5),
                   ('High', 0, 0.2)]
    f1_scores = []
    model_names = []
    groups = []
    for item in test_ranges:
        group_df = pd.concat([(data if (((data['final_score'] <= 2).sum() >= (item[1] * len(data))) & (
                (data['final_score'] <= 2).sum() <= item[2] * len(data))) else pd.DataFrame()) for _, data in g])
        user_group = pd.unique(group_df['user_id'])

        important_users = all_results.loc[all_results['user_id'].isin(user_group)]

        for model_title in list(model_preds.keys()):
            if model_title != 'true':
                model_names.append(model_title)
                f1_scores.append(f1_score(important_users['true'], important_users[model_title], average='weighted'))
                groups.append(item[0])

    data = pd.DataFrame({"Weighted F1-Score": f1_scores,
                         'Percentage of Low Grades': groups,
                         'Model': model_names})

    plot = sns.barplot(data=data, x="Percentage of Low Grades", y='Weighted F1-Score', hue='Model')
    for item in plot.get_xticklabels():
        item.set_fontweight('bold')
    plt.xlabel('Student Performance Level', fontweight='bold')
    plt.ylabel('Weighted F1-Score', fontweight='bold')
    plt.ylim([0.5, 1])

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    cm_array = np.array(cm)

    group_counts = ["{0:0.0f}".format(value) for value in
                    cm_array.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm_array.flatten() / np.sum(cm_array)]
    labels = [f"{v2}\n{v3}" for v2, v3 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(5, 5)
    sns.heatmap(cm_array, annot=labels, fmt='', cmap='Blues')
    plt.title(f'Confusion matrix for {model_name}')
    plt.ylabel('Actual Classes', fontweight='bold')
    plt.xlabel('Predicted Classes', fontweight='bold')

    plt.show()


model_preds = {}
for dataset_version in ['base', 'node2vec', 'Deepwalk']:
    X_train, X_test, y_train, y_test = split_dataset(
        mooc_df if dataset_version == 'base' else mooc_with_node2vec if dataset_version == 'node2vec' else mooc_with_deepwalk)

    model_preds['true'] = y_test
    model_title = model_name if dataset_version == 'base' else model_name + ' + ' + dataset_version
    model = pickle.load(open(f'./models/{dataset_version}_{model_name}_classifier.sav', 'rb'))
    y_pred = model.predict(X_test)
    model_preds[model_title] = y_pred
    plot_confusion_matrix(y_test, y_pred, model_title)
    compute_feature_importance(X_train, y_train, model, model_title)

X_train, X_test, y_train, y_test = split_dataset(mooc_df)
all_results = pd.DataFrame(model_preds)
all_results = pd.concat([X_test[['user_id']], all_results], axis=1)
showcase_user_categories(X_train[['user_id']], y_train, all_results)
