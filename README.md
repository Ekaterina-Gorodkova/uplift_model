# X5 Uplift Modeling

Репозиторий для EDA, feature engineering и baseline uplift-моделирования на датасете X5

## Структура
- `data/raw/` — исходные CSV-файлы
- `data/processed/` — подготовленные таблицы и feature store
- `notebooks/01_eda.ipynb` — EDA с акцентом на treatment/control
- `notebooks/02_feature_engineering.ipynb` — построение признаков
- `notebooks/03_uplift_modeling.ipynb` — baseline uplift modeling
- `src/data_processing.py` — чтение и валидация сырых данных
- `src/eda.py` — функции для EDA
- `src/feature_engineering.py` — построение признаков, включая chunk processing
- `src/uplift_models.py` — baseline uplift pipeline
- `src/evaluation.py` — uplift-метрики и кривые
- `configs/config.yaml` — конфиг путей и имен колонок
