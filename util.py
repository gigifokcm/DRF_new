"""Functions used in both the training and prediction parts"""
import pandas as pd
import numpy as np
import seaborn as sns
import re
import pickle
import datetime
import h2o
import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# from gensim.models import KeyedVectors

# --------------------------------------------------------------------------------------
# Commonly used functions
# --------------------------------------------------------------------------------------

def time_now():
    """
    :return: time now in this format: '%Y%m%d %H:%M:%S'.
    """
    return datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')


def format_input(train_df, test_df, id_label, target_label, train_label=""):
    """
    Split the train_df and test_df into DataFrames.
    :param train_df:
    :param test_df:
    :param id_label: the unique key for the entry
    :param target_label: column names for the target
    :param train_label: column names for the required label in train and test
            if blank, the function assumes all columns except for id_label and test_label are included in train and test
    :return: rain_df, test_df, target, train_id, test_id, train_label
    """
    if train_label == "":
        train_label = [e for e in list(train_df) if e not in (target_label+id_label)]

    target = train_df[target_label]

    train = train_df[train_label]
    test = test_df[train_label]

    train_id = train_df[id_label]
    test_id = test_df[id_label]

    return train, test, target, train_id, test_id, train_label


def dist(df):
    """
    This is a simple analysis on data by count. Skip all 'object' type variables.
    Use matplotlib.pyplot.show() to show the graph.
    :param df: dataframe
    :return: print the distribution by count and by % with a graph for each column in df
    """

    df_new = df.select_dtypes(['object', 'integer'])

    for column in df_new:
        temp = df_new[column].value_counts()
        print('The target distribution by count:(' + column + ")")
        print(temp)
        print('The target distribution by %:('+ column + ")")
        temp_pc = temp/df_new.shape[0]
        print(temp_pc.map(lambda n: '{:,.2%}'.format(n)))


def encoding(df, encode_col=None):
    """
    Label encoding for columns with 2 categorical values
    Perform one-hot encoding for columns with more than categorical values
    :param df: DataFrame (can include numerical values but the function will skip those columns)
    :param encode_col: customise the columns for encoding
    :return: DataFrame; print the number of columns encoded
    """
    le_count = 0
    ohe_count = 0

    if encode_col is None or encode_col == "":
        cat_data_label = list(df.select_dtypes('object'))
    else:
        cat_data_label = encode_col

    if cat_data_label == []:
        print("The DataFrame contains no object items for encoding.")

    else:
        df_cat = df[cat_data_label].apply(pd.Series.nunique, dropna=False, axis=0)

        for s in cat_data_label:
            df[s] = df[s].astype(str)

            if df_cat[s] >= 3:
                # use pd.concat to join the new columns with your original DataFrame
                temp = pd.concat([df, pd.get_dummies(df[s], prefix=s)], axis=1)
                df = temp

                # now drop the original 'country' column (you don't need it anymore)
                df.drop([s], axis=1, inplace=True)
                ohe_count += 1
            else:
                le = LabelEncoder()
                le.fit(df[s])
                df[s] = le.transform(df[s].astype(str))
                le_count += 1
                le = None
        print(le_count, "columns are label encoding and", str(ohe_count), "columns are one-hot encoded.")
    return df


def missing_values_table(df, prefix):
    """
    This provides a summary of missing data by count and % for each column
    :param df:
    :param prefix:
    :return:
    """

    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: prefix + 'Missing_Values', 1: prefix + 'Missing_%_Total'})

    # Print some summary information
    print("This dataframe has " +
           str(np.array(mis_val_table_ren_columns[prefix+'Missing_Values'] > 0).sum()) +
          ' columns with missing values out of ' + str(df.shape[1]) + " columns.")

    return mis_val_table_ren_columns


def write_input(folder, train_df, test_df, suffix=""):
    train_df.to_csv(os.path.join(folder, ('train' + str(suffix)+'.csv')), sep=",", index=False)
    test_df.to_csv(os.path.join(folder, ('test' + str(suffix)+'.csv')), sep=",", index=False)


def write_results_main(main_df, train_size, path, suffix=""):
    train = main_df[main_df.index < train_size]
    test = main_df[main_df.index >= train_size]
    write_input(folder=path, train_df=train, test_df=test, suffix=suffix)


def read_input(folder, suffix=""):
    train_df = pd.read_csv(os.path.join(folder, ('train' + str(suffix)+'.csv')))
    test_df = pd.read_csv(os.path.join(folder, ('test' + str(suffix)+'.csv')))
    target_df = pd.read_csv(os.path.join(folder, 'target.csv'))

    return train_df, target_df, test_df


def eda_pack(train, test, index, run_missing_data=1, run_predictor_ana=1):
    # Create a DataFrame called EDA summarising some key features of the test and test data
    eda_df = pd.DataFrame(index=index)

    #  region Missing data analysis
    if run_missing_data == 1:
        print('[{}] Missing data analysis starts'.format(time_now()))
        print("Train data - ")
        missing_values_train = pd.DataFrame(missing_values_table(train, "Train_"))
        print("Test data - ")
        missing_values_test = pd.DataFrame(missing_values_table(test, "Test_"))
        eda_df = pd.concat(objs=[eda_df, missing_values_train, missing_values_test], axis=1, sort=False)

        print("Missing value statistics for train data (%):")
        print(missing_values_train.sort_values(by='Train_Missing_%_Total', ascending=False)['Train_Missing_%_Total']
              .map(lambda x: '{:.2f}'.format(x)).head(10))
    # endregion

    # region Predictor analysis
    if run_predictor_ana == 1:
        print('[{}] Predictors analysis starts'.format(time_now()))
        print("Data type of predictors - value counts:")
        print(train.dtypes.value_counts())

        col_header = list(pd.DataFrame([1]).describe().index)
        train_info = train.describe().transpose().rename(
            columns=dict(zip(col_header, list(map(lambda x: "Train_" + x, col_header)))))

        test_info = test.describe().transpose().rename(
            columns=dict(zip(col_header, list(map(lambda x: "Test_" + x, col_header)))))

        eda_df = pd.concat([eda_df, train_info, test_info], sort=False, axis=1)

        # Class_count (to ensure that the train and test data aligns)
        eda_df['class_count_train'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique,
                                                                                     dropna=False,
                                                                                     axis=0)
        eda_df['class_count_test'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique,
                                                                                    dropna=False,
                                                                                    axis=0)
    # endregion

    # main_df['Flag_TrainTest'] = np.where(main_df.index < train_size, "Train", "Test")
    main_df = pd.concat([train, test], sort=False, ignore_index=True)

    # region Train and test data statistics
    if run_predictor_ana == 1:
        # Number of unique classes in each object column
        eda_df['class_count_main'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique,
                                                                                    dropna=False,
                                                                                    axis=0)
        eda_df['class_count_main_noNaN'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique, axis=0)
        eda_df['dtypes'] = main_df.dtypes
    # endregion

    return eda_df



def pairgrid(df):
    """
    plot pairgrid (histogram in diagonal, scatterplot in upper triangle and kde for lower triangle.
    (use matplotlib.pyplot.show() to show the graph).
    :param df: dataframe
    :return: graph
    """

    g = sns.PairGrid(df, hue="Flag_TrainTest", palette='rainbow')
    g.map_diag(plt.hist)
    g.map_upper(plt.scatter)
    g.map_lower(sns.kdeplot)


def save_model(model, path):
    """ Saving the neural net model to model_path
    """
    model_path = h2o.save_model(model=model, path=path, force=True)

    return model_path


def pca_analysis(x):
    pca = PCA()
    fit = pca.fit(x)
    pc = pca.transform(x)

    print(("Explained Variance (first 5 components: %s") % fit.explained_variance_ratio_[0:5])

    acc = pd.DataFrame(fit.explained_variance_ratio_).cumsum()

    plt.figure(1, figsize=(14, 13))
    plt.axes([.2, .2, .7, .7])
    plt.plot(acc, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('acc_explained_variance_ratio_')

    return pca, pc


def pc_3d(x_pca, y):
    # Plot initialisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_pca['PCA0'], x_pca['PCA1'], x_pca['PCA2'], c=y, cmap="Set2_r", s=60)

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA on the iris data set")


def load_trained_model(model_path):
    """ Loading the neural net model from model_path
    """
    model = load_model(model_path)
    return model


def stdout_off():
    sys.stdout = open(os.devnull, "w")


def stdout_on():
    sys.stdout = sys.__stdout__


def plotgraph(df, hue):
    print ("GraphPlot initialises")

    for column in df:
        if df[column].dtypes == 'object':
            sns.countplot(x=column, data=df, hue=hue, palette="Greens_d")
        else:
            g = sns.FacetGrid(df[[column,hue]], hue=hue)
            g.map(sns.distplot, column, label=hue)
            g.add_legend()


def variable_importance_graph(model):
    fig, ax = plt.subplots()
    variables = model._model_json['output']['variable_importances']['variable']
    y_pos = np.arange(len(variables))
    scaled_importance = model._model_json['output']['variable_importances']['scaled_importance']
    ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('Scaled Importance')
    ax.set_title('Variable Importance')
    plt.show()

# --------------------------------------------------------------------------------------
# Program specific functions - Home Credit
# --------------------------------------------------------------------------------------



def clean_train_test(df):
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])

    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)

    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)

    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())

    df['NEW_SCORES_STD'] = np.max(
        np.abs(np.array(df['EXT_SOURCE_1'] - df['EXT_SOURCE_2'])),
        np.abs(np.array(df['EXT_SOURCE_2'] - df['EXT_SOURCE_3'])))

    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    return df