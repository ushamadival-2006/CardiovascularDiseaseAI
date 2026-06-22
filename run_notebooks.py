import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebooks = [
    'notebooks/02_preprocessing.ipynb',
    'notebooks/03_feature_selection_and_train_test_split.ipynb',
    'notebooks/04_model_training.ipynb'
]

for nb_path in notebooks:
    print(f'Executing {nb_path}')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Completed {nb_path}')
