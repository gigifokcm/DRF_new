# --------------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

import settings.para_pre as para
import util

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('[{}] Program starts'.format(util.time_now()), "(" + str(para.PARA_FUNCTION) + "_" + str(para.VERSION) + ")")

    # region Reading data - target, test and their settings and formatting to DataFrames
    print('[{}] Reading data from: '.format(util.time_now()), para.RAW_INPUT_PATH)
    train_df = pd.read_csv(os.path.join(para.RAW_INPUT_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(para.RAW_INPUT_PATH, 'test.csv'))

    print('Training data shape (original): ', train_df.shape)
    print('Test data shape: (original)', test_df.shape)

    train, test, target, train_id, test_id, train_label = util.format_input(train_df=train_df,
                                                                            test_df=test_df,
                                                                            id_label=para.ID_LABEL,
                                                                            target_label=para.TARGET_LABEL,
                                                                            train_label=para.TRAIN_LABEL)
    # endregion

    # region Data cleaning
    print('[{}] Data cleaning starts'.format(util.time_now()))
    print("Cleaning data: skipped")
    # endregion

    # region Summarising data from target
    if para.RUN_TARGET_ANA ==1:
        print('[{}] Analyzing target'.format(util.time_now()))
        util.dist(target)
    # endregion

    # Create a DataFrame called EDA summarising some key features of the test and test data
    EDA_df = pd.DataFrame()
    EDA_df.reset_index()
    EDA_df.reindex(index=train_label)

    #  region Missing data analysis
    if para.RUN_MISSING_DATA == 1:
        print('[{}] Missing data analysis starts'.format(util.time_now()))
        print("Train data - ")
        missing_values_train = pd.DataFrame(util.missing_values_table(train, "Train_"))
        print("Test data - ")
        missing_values_test = pd.DataFrame(util.missing_values_table(test, "Test_"))
        EDA_df = pd.concat(objs=[EDA_df, missing_values_train, missing_values_test], axis=1, sort=False)

        print("Missing value statistics for train data (%):")
        print(missing_values_train.sort_values(by='Train_Missing_%_Total', ascending=False)['Train_Missing_%_Total']
              .map(lambda x: '{:.2f}'.format(x)).head(10))
    # endregion

    # region Predictor analysis
    if para.RUN_PREDICTOR_ANA == 1:
        print('[{}] Predictors analysis starts'.format(util.time_now()))
        print("Data type of predictors - value counts:")
        print(train.dtypes.value_counts())

        col_header = list(pd.DataFrame([1]).describe().index)
        train_info = train.describe().transpose().rename(
            columns=dict(zip(col_header, list(map(lambda x: "Train_" + x, col_header)))))

        test_info = test.describe().transpose().rename(
            columns=dict(zip(col_header, list(map(lambda x: "Test_" + x, col_header)))))

        EDA_df = pd.concat([EDA_df, train_info, test_info], sort=False, axis=1)

        # Class_count (to ensure that the train and test data aligns)
        EDA_df['class_count_train'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique,
                                                                                        dropna=False,
                                                                                        axis=0)
        EDA_df['class_count_test'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique,
                                                                                       dropna=False,
                                                                                       axis=0)
    # endregion

    train_size = train.shape[0]
    main_df = pd.concat([train, test], sort=False, ignore_index=True)
    # main_df['Flag_TrainTest'] = np.where(main_df.index < train_size, "Train", "Test")

    # region Train and test data statistics
    if para.RUN_PREDICTOR_ANA == 1:
        # Number of unique classes in each object column
        EDA_df['class_count_main'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique,
                                                                                       dropna=False,
                                                                                       axis=0)
        EDA_df['class_count_main_noNaN'] = train.select_dtypes(['object', 'int64']).apply(pd.Series.nunique, axis=0)
        EDA_df['dtypes'] = main_df.dtypes
    # endregion

    # Print EDA
    print('[{}] Printing data analysis to: '.format(util.time_now()), para.INPUT_PATH)
    if EDA_df.shape[0] > 0:
        EDA_df.to_csv(os.path.join(para.INPUT_PATH, "EDA.csv"), sep=",", index=True)


    # region Encoding target
    print('[{}] Encoding starts (target)'.format(util.time_now()))
    target = util.encoding(target, para.TARGET_ENCODING)
    target_label = list(target)
    # endregion

    # region Print train_id, test_id and target
    print('[{}] Printing formatted results to: '.format(util.time_now()), para.INPUT_PATH)
    train_id.to_csv(os.path.join(para.INPUT_PATH, "train_id.csv"), index=False)
    test_id.to_csv(os.path.join(para.INPUT_PATH, "test_id.csv"), index=False)
    target.to_csv(os.path.join(para.INPUT_PATH, "target.csv"), index=False)
    # util.write_input(para.INPUT_PATH, train, target, test, "")
    # endregion

    # region Encoding predictors
    print('[{}] Encoding starts (train)'.format(util.time_now()))
    if para.RUN_NORMALISATION == 1:
        main_encoded = util.encoding(main_df, para.PREDICTOR_ENCODING)

        a = pd.read_csv(os.path.join(para.INPUT_PATH, "train_encoded.csv"))

        # Write results
        print('[{}] Printing formatted results to: '.format(util.time_now()), para.INPUT_PATH)
        util.write_results_main(main_encoded, train_size, para.INPUT_PATH, "_encoded")
        main_df = main_encoded

    # endregion

    # region Imputer
    if para.RUN_IMPUTATION == 1:
        print('[{}] Imputing missing values'.format(util.time_now()))

        imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)

        main_imputed = pd.DataFrame(imputer.fit_transform(main_df),
                                    columns=main_df.columns,
                                    index=main_df.index)

        print('[{}] Printing formatted results to: '.format(util.time_now()), para.INPUT_PATH)
        util.write_results_main(main_imputed, train_size, para.INPUT_PATH, "_imputed")
        main_df = main_imputed
    # endregion

    # region Normalisation
    if para.RUN_NORMALISATION == 1:
        print('[{}] Normalisation starts'.format(util.time_now()))

        main_std = pd.DataFrame(StandardScaler().fit_transform(main_df),
                                columns=main_df.columns,
                                index=main_df.index)

        print('[{}] Printing formatted results to: '.format(util.time_now()), para.INPUT_PATH)
        util.write_results_main(main_std, train_size, para.INPUT_PATH, "_std")
        main_df = main_std
    # endregion

    print('[{}] Program ended'.format(util.time_now()))

    print('[{}] Displaying graphs'.format(util.time_now()))
    plt.show()




