import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Загрузка данных
df = pd.read_csv('merged_for_boosting.csv')
features = ['max', 'mean', 'std', 'perc_90', 'num_passed', 'range']
X = df[features]
y = df['rel']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Модель с параметрами для малых данных и дисбаланса
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    num_leaves=15,                 # небольшое дерево
    min_child_samples=5,           # главное: уменьшаем с 20 до 5
    min_child_weight=0.001,
    min_gain_to_split=0.0,
    min_data_in_leaf=5,            # тоже уменьшаем
    scale_pos_weight=746/154,      # балансировка весов (обратное отношение классов)
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=1
)

# Обучение с валидацией
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(10)]
)

import pickle
with open('model.pkl', 'wb') as f:
	pickle.dump(model, f)

# Оценка
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"\nTest AUC: {auc:.4f}")

# Важность признаков
print("\nFeature importance:")
for feat, imp in zip(features, model.feature_importances_):
    print(f"{feat}: {imp}")