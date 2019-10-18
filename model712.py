import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains columns used for training (features)
    as well as the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training set: X and y together
    with the preprocessed evaluation set: X_eval.

    Make sure you return X_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]

    For y you can either return a pd.DataFrame with one column or pd.Series.

    :param df: the dataset
    :type df: pd.DataFrame
    :return: X, y, X_eval
    """

    # Extract major and minor categories
    pattern = r'(?<="slug":")(\w*\s*.*)(?=")'
    df['short_category'] = df.category.str.extract(pattern)
    df[['major_category', 'minor_category']] = df.short_category.str.split('/', n=0, expand=True)

    # Update goal values to USD for non US countries
    df['country'] = np.where((df['country'] != 'US') & (df['country'] != 'GB'), 'Other', df['country'])
    df["goal_usd"] = df["goal"] * df["static_usd_rate"]
    df["goal_usd"] = df["goal_usd"].astype(int)

    # Add features
    df['blurb_length'] = df.blurb.str.len()
    df['name_length'] = df.name.str.len()
    df['slug_length'] = df.slug.str.len()

    # Launch Day / Month
    df['launch_dt'] = pd.to_datetime(df['launched_at'], unit='s')
    df['launch_month'] = pd.DatetimeIndex(df['launch_dt']).month
    df['launch_year'] = pd.DatetimeIndex(df['launch_dt']).year
    df['launch_day'] = pd.DatetimeIndex(df['launch_dt']).day

    # calculate duration of fundraising and time to launch
    unix_seconds_per_day = 86400
    df['duration'] = df.deadline - df.launched_at
    df['duration'] = df['duration'].div(unix_seconds_per_day).abs().astype(int)
    df['time_to_launch'] = df.launched_at - df.created_at
    df['time_to_launch'] = df['time_to_launch'].div(unix_seconds_per_day).abs().astype(int)

    # change categorical features in type category (for Decision Tree)
    categorical_columns = ['minor_category', 'country']
    for col in categorical_columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    # Drop columns not required and fillna with 0
    columns_to_drop = ['photo', 'name', 'blurb', 'slug', 'currency_symbol', 'creator', 'profile', 'urls',
                       'source_url', 'short_category', 'category', 'goal', 'disable_communication',
                       'deadline', 'created_at', 'location', 'launched_at', 'static_usd_rate',
                       'currency', 'currency_trailing_code', 'slug_length', 'country', 'launch_dt',
                       'friends', 'is_starred', 'is_backing', 'name_length', 'slug_length',
                       'permissions', 'blurb_length', 'major_category']
    df.drop(columns_to_drop, axis=1, inplace=True)
    df.fillna(0, inplace=True)

    # save labels to know what rows are in evaluation set
    # evaluation_set is a boolean so we can use it as mask

    msk_eval = df.evaluation_set
    X = df[~msk_eval].drop(["state"], axis=1)
    y = df[~msk_eval]["state"]
    X_eval = df[msk_eval].drop(["state"], axis=1)

    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """
    params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=5)
    model.fit(X, y)
    return model.best_estimator_


def predict(model, X_test):
    """This functions takes your trained model as well
    as a processed test dataset and returns predictions.

    On KATE, the processed test dataset will be the X_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.Series

    :param model: your trained model
    :param X_test: a processed test set (on KATE it will be X_eval)
    :return: y_pred, your predictions
    """

    y_pred = model.predict(X_test)
    return pd.Series(y_pred)
