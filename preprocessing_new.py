# --------------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------------
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import gc

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

    num_rows = 10000 if para.DEBUG else None
    train_df = pd.read_csv(os.path.join(para.RAW_INPUT_PATH, 'train.csv'), num_rows = num_rows)
    test_df = pd.read_csv(os.path.join(para.RAW_INPUT_PATH, 'test.csv'), num_rows = num_rows)

    print('(from raw data) Training data shape is {} and test data shape is {}.'.format(train_df.shape, test_df.shape))

    train, test, target, train_id, test_id, train_label = util.format_input(train_df=train_df,
                                                                            test_df=test_df,
                                                                            id_label=para.ID_LABEL,
                                                                            target_label=para.TARGET_LABEL,
                                                                            train_label=para.TRAIN_LABEL)

    train_size = train.shape[0]
    main_df = pd.concat([train, test], sort=False, ignore_index=True)
    del train, test
    gc.collect()
    # endregion

    # region Data cleaning
    print('[{}] Data cleaning starts'.format(util.time_now()))
    if para.RUN_DATACLEAN == 1:
        util.clean_train_test(main_df)
    # endregion

    # region Summarising data from target
    if para.RUN_TARGET_ANA == 1:
        print('[{}] Analyzing target'.format(util.time_now()))
        util.dist(target)
    # endregion

    # EDA analysis
    EDA_df = util.eda_pack(train, test, train_label, para.RUN_MISSING_DATA, para.RUN_PREDICTOR_ANA)

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




