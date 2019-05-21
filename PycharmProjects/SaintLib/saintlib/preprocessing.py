import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def choosing_features(X, y, data_type="num_only", cardinal_depth = 10, split_train_test=False):
    if split_train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y)

    result = None
    if data_type is "num_only":
        result = X.select_dtypes(include=[np.number])
    if data_type is "all_onehot":
        print("choosing_features. all_onehot")
        #One-hot Encoding이 가능한 칼럼 선별
        cardinality_col = [col for col in X.columns if
                        X[col].nunique()<cardinal_depth and
                       X[col].dtype =='object']
        numeric_cols = [col for col in X.columns if
                        X[col].dtype in ['int64', 'float64']]
        final_cols = cardinality_col + numeric_cols
        result = pd.get_dummies(X[final_cols])
    return result



def echo(param):
    """
    saintlib이 정상적으로 로딩되었는지 확인하는 메소드
    from os import path
    import sys
    sys.path.append(path.abspath('/home/salgugol/ML/github/PycharmProjects/SaintLib'))
    import saintlib.preprocessing as saint_pre
    :param param:for echo
    :return: param
    """
    print("saintlib preprocessing")
    print("Param is %s" %param)
    return param


def handling_null_data(data, how_null="drop"):
    if how_null is "drop":
        data = data.dropna(axis=0)
    return data


def pick_cardinal_cols(str_train_data, str_test_data, str_target_col, str_column_to_drop, n_one_hot_cardinal):
    train_data = pd.read_csv(str_train_data)
    test_data = pd.read_csv(str_test_data)

    #STEP1. # 타겟 값이 없는 데이터 삭제
    train_data.dropna(axis=0, subset=[str_target_col], inplace=True)
    train_y = train_data[str_target_col]
    #print(train_data.head())

    #STEP2. null이 들어 있는 column & 전달인자로 받은 칼럼 삭제
    # TODO: 개선을 위해서는 아래처럼 단순히 드랍시키는 것은 좋지 않다.
    # https://www.kaggle.com/dansbecker/handling-missing-values
    cols_with_na = [col for col in train_data.columns
                    if train_data[col].isnull().any()]
    candidate_train_predictors = train_data.drop([str_target_col]+[str_column_to_drop] + cols_with_na, axis=1)
    candidate_test_predictors = test_data.drop([str_column_to_drop]+ cols_with_na, axis=1)  # test용에는 "SalePrice"를 남겨둠

    #STEP3. One-hot Encoding이 가능한 칼럼 선별
    # 최종 칼럼(final_cols) = low_cardinality_cols + numeric_cols
    low_cardinality_cols =  [col for col in candidate_train_predictors.columns if
                        candidate_train_predictors[col].nunique()<n_one_hot_cardinal and
                       candidate_train_predictors[col].dtype =='object']

    numeric_cols = [col for col in candidate_train_predictors.columns if
                    candidate_train_predictors[col].dtype in ['int64', 'float64']]
    final_cols = low_cardinality_cols + numeric_cols

    train_predictors = candidate_train_predictors[final_cols]
    test_predictors = candidate_test_predictors[final_cols]

    return train_predictors, test_predictors, train_y


def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring = 'neg_mean_absolute_error').mean()

