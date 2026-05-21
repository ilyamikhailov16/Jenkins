from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

DATASET_PATH = "./possum.csv"
DROP_NULLS = True
DROP_DUPLICATED = True
COLS_CONVERSION = {
    "cols_to_delete": ["case"],
    "cols_to_float": ["age", "foot_length"],
    "cols_to_str": ["site"],
}
TARGET_COL = "age"
RANDOM_STATE = 42
DATA_FOLDER = "./data/"

X_TRAIN_PATH = DATA_FOLDER + "X_train.csv"
X_TEST_PATH = DATA_FOLDER + "X_test.csv"
Y_TRAIN_PATH = DATA_FOLDER + "y_train.csv"
Y_TEST_PATH = DATA_FOLDER + "y_test.csv"
MODEL = GradientBoostingRegressor
METRIC_FUNC = r2_score