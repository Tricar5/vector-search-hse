import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('merged_for_boosting.csv')
features = ['idx', 'max', 'mean', 'std', 'perc_90', 'num_passed', 'range']
X = df[features]
y = df['rel']

print("=== Проверка данных ===")
print("NaN в X:", X.isnull().sum().sum())
print("Inf в X:", np.isinf(X).sum().sum())
print("Уникальные y:", y.unique())
print("Баланс классов:\n", y.value_counts())

print("\n=== Корреляция признаков с y (точечно-бисериальная) ===")
for col in X.columns:
    corr = X[col].corr(y)
    print(f"{col}: {corr:.4f}")

print("\n=== Простой решающий пень (depth=1) через DecisionTreeClassifier ===")
dt = DecisionTreeClassifier(max_depth=1, random_state=42)
dt.fit(X, y)
print("Сплит найден:", dt.tree_.feature[0] != -2)
if dt.tree_.feature[0] != -2:
    print(f"Признак разбиения: {X.columns[dt.tree_.feature[0]]}, порог: {dt.tree_.threshold[0]:.4f}")
    print("Gini impurity на корне:", dt.tree_.impurity[0])
    print("Gini в левом листе:", dt.tree_.impurity[1])
    print("Gini в правом листе:", dt.tree_.impurity[2])
else:
    print("Дерево не сделало ни одного разбиения!")

print("\n=== Random Forest (3-fold CV AUC) ===")
rf = RandomForestClassifier(n_estimators=10, random_state=42)
scores = cross_val_score(rf, X, y, cv=3, scoring='roc_auc')
print(f"RF AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\n=== LightGBM с максимально свободными параметрами ===")
params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'num_leaves': 2,
    'min_child_samples': 1,
    'min_gain_to_split': 0.0,
    'max_depth': 1,
    'learning_rate': 1.0,
    'num_iterations': 1,
    'min_data_in_leaf': 1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'verbose': -1
}
train_data = lgb.Dataset(X, label=y)
try:
    model = lgb.train(params, train_data, num_boost_round=1)
    tree_info = model.dump_model()['tree_info'][0]
    if 'split_index' in tree_info['tree_structure']:
        print("LightGBM сделал разбиение!")
        print(tree_info['tree_structure'])
    else:
        print("LightGBM не сделал разбиения: дерево состоит из одного листа")
        print("Значение в листе:", tree_info['tree_structure']['leaf_value'])
except Exception as e:
    print("Ошибка при обучении:", e)