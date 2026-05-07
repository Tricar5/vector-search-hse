import os
import numpy as np
import pandas as pd

def conv(x):
    if x == b'True' or x == 'True':
        return 1
    if x == b'False' or x == 'False':
        return 0
    return float(x) if '.' in str(x) else int(x)

# Список всех CSV-файлов (исключая скрипты .py)
files = [f for f in os.listdir('.') if not f.endswith('.py') and os.path.isfile(f)]

all_dfs = []
for file in files:
    try:
        # Читаем CSV через pandas (проще для конкатенации)
        df = pd.read_csv(file, skiprows=1, names=['idx','max','mean','std','perc_90','num_passed','density','range','rel'],
                         converters={col: conv for col in range(9)})
        # Добавляем колонку с именем файла (без расширения, можно и полное)
        all_dfs.append(df)
        print(f"Загружен {file}: {len(df)} строк")
    except Exception as e:
        print(f"Ошибка в {file}: {e}")

# Объединяем все DataFrame
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\nИтоговый датасет: {len(merged)} строк, колонки: {list(merged.columns)}")
    
    # Сохраняем в один CSV для обучения
    merged.to_csv('merged_for_boosting.csv', index=False)
    print("Сохранён файл merged_for_boosting.csv")
    
    # Дополнительно: выводим статистику по целевой переменной
    print("\nРаспределение rel:")
    print(merged['rel'].value_counts(normalize=True))
else:
    print("Нет подходящих файлов.")