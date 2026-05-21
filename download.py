import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from config import *


def load(path: str) -> pl.DataFrame:
    """Загрузка данных с мгновенной обработкой пропусков."""

    # Загрузка данных с корретными null значениями
    null_names = ["NA", "Nan", "NAN", "None", "NULL", "null", "Null", "NaN"]
    df = pl.read_csv(path, null_values=null_names)
    df = df.with_columns(cs.float().fill_nan(None))

    return df


def cols_preprocess(
    df: pl.DataFrame, cols_conversion: dict[str, list[str]]
) -> pl.DataFrame:
    """Конверсия типов и удаление лишних столбцов за один проход."""

    # Удаление лишних столбцов
    df = df.drop(cols_conversion.get("cols_to_delete", []))

    # Конверсия типов
    dtypes_map = {}
    for col in cols_conversion.get("cols_to_int", []):
        dtypes_map[col] = pl.Int64
    for col in cols_conversion.get("cols_to_float", []):
        dtypes_map[col] = pl.Float64
    for col in cols_conversion.get("cols_to_str", []):
        dtypes_map[col] = pl.String

    if dtypes_map:
        df = df.cast(dtypes_map)

    return df


def rows_preprocess(
    df: pl.DataFrame, drop_nulls: bool = True, drop_duplicated: bool = True
) -> pl.DataFrame:
    """Удаление строк с null и дубликатов."""

    # Удаление строк с пустымии значениями
    if drop_nulls:
        df = df.drop_nulls()
    else:
        # Пакетное заполнение пропусков по типам данных
        df = df.with_columns(cs.float().fill_null(0.0), cs.string().fill_null("null"))

    if drop_duplicated:
        df = df.unique(maintain_order=True)

    return df


def cat_preprocess(df: pl.DataFrame) -> pl.DataFrame:
    cat_cols = df.select(cs.string()).columns

    if not cat_cols:
        return df

    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(df.select(cat_cols))
    new_col_names = encoder.get_feature_names_out(cat_cols)
    df_encoded = pl.DataFrame(encoded_array, schema=list(new_col_names))

    return pl.concat([df.drop(cat_cols), df_encoded], how="horizontal")


def preprocess(
    df: pl.DataFrame,
    cols_conversion: dict[str, list[str]],
    drop_nulls: bool = True,
    drop_duplicated: bool = True,
) -> pl.DataFrame:
    df = (
        df.pipe(cols_preprocess, cols_conversion)
        .pipe(rows_preprocess, drop_nulls, drop_duplicated)
        .pipe(cat_preprocess)
    )
    return df


def split(df: pl.DataFrame, y: str) -> tuple[pl.DataFrame]:
    X = df.select(pl.exclude(y))
    y = df.select(pl.col(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def scale(
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    scaler,
) -> tuple[pl.DataFrame]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler(), num_features),
            ("cat", "passthrough", cat_features),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="polars")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def optimize_series_dtype(series: pl.Series) -> pl.Series:
    if series.dtype in (pl.Float32, pl.Float64):
        values = series.to_numpy()

        is_integer_like = np.all(np.isfinite(values)) and np.allclose(
            values,
            np.round(values),
        )
        if is_integer_like:
            return series.cast(pl.Int64).shrink_dtype()

    return series.shrink_dtype()


def memory_optimize(
    X_train: pl.DataFrame, X_test: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    X_train = pl.DataFrame([optimize_series_dtype(s) for s in X_train])
    X_test = pl.DataFrame([optimize_series_dtype(s) for s in X_test])
    return X_train, X_test


if __name__ == "__main__":
    pl.set_random_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = load(DATASET_PATH)
    preprocessed_df = preprocess(df, COLS_CONVERSION, DROP_NULLS, DROP_DUPLICATED)
    X_train, X_test, y_train, y_test = split(preprocessed_df, y=TARGET_COL)
    X_train, X_test = scale(X_train, X_test, num_features=X_train.columns[:9], cat_features=X_train.columns[9:], scaler=StandardScaler)
    X_train, X_test = memory_optimize(X_train, X_test)

    # Сохраняем выборки в CSV
    X_train.write_csv(X_TRAIN_PATH)
    X_test.write_csv(X_TEST_PATH)
    y_train.write_csv(Y_TRAIN_PATH)
    y_test.write_csv(Y_TEST_PATH)

    # pl.Config.set_tbl_cols(-1)       # показывать все столбцы
    # pl.Config.set_tbl_width_chars(0) # без ограничения по ширине
    # print(X_train.head())