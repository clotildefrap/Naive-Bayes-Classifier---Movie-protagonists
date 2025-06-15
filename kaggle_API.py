import kagglehub

# Download latest version
path = kagglehub.dataset_download("gufukuro/movie-scripts-corpus")

print("Path to dataset files:", path)

#Path to dataset files: /home/clotilde/.cache/kagglehub/datasets/gufukuro/movie-scripts-corpus/versions/1

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
