# --------------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------------
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.metrics import roc_auc_score
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
import gc

import settings.para as para
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

    # initiate h2o
    h2o.init(port=5432)

    # region Reading data - target, test and their settings and formatting to DataFrames
    print('[{}] Loading data from: '.format(util.time_now()), para.RESOURCE_PATH)
    print('Data set with suffix:', para.INPUT_SUFFIX)

    # Set data type (skipped)

    train_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'train' + para.INPUT_SUFFIX + '.csv'))
    test_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'test' + para.INPUT_SUFFIX + '.csv'))
    target = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'target.csv'))

    print('Training data shape (original): ', train_df.shape)
    print('Test data shape: (original)', test_df.shape)
    print('Target data shape: (original)', target.shape)
    # endregion

    #bureau_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'bureau.csv'))
    #bureau_bal_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'bureau_balance.csv'))
    #cc_balance_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'credit_card_balance.csv'))
    #installment_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'installments_payments.csv'))
    #cash_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'POS_CASH_balance.csv'))
    #previous_df = h2o.import_file(os.path.join(para.RESOURCE_PATH, 'previous_application.csv'))

    # region train validate split + select variables
    train_with_target = train_df.cbind(target)
    print("train_with_target: ", train_with_target.shape)

    if para.TRAIN_LABEL == "":
        train_label = train_df.col_names
    else:
        train_label = para.TRAIN_LABEL

    train, test = train_with_target.split_frame(ratios=[1-para.TEST_SIZE])
    # endregion

    # region PCA
    if para.RUN_PCA == 1:
        pca_decomp = H2OPrincipalComponentAnalysisEstimator(k=124,
                                                            transform="Standardize",
                                                            pca_method="Power",
                                                            impute_missing=True)

        pca_decomp.train(training_frame=train_df)
        pred = pca_decomp.predict(train_df)
    # endregion

    # region Model
    print('[{}] Model training starts'.format(util.time_now()))

    # Define model
    model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)

    # Train model
    model.train(x=train_label, y='TARGET', training_frame=train_with_target)
    # endregion

    # region Predictions + evaluations
    print('[{}] Finished model training. Prediction and model evaluation start'.format(util.time_now()))

    # Validation
    pred = model.predict(test_data=test)
    auc = roc_auc_score(y_true=test[:, 'TARGET'].as_data_frame().values,
                        y_score=pred.as_data_frame().values)

    print("The validation AUC is:", auc)
    print(model.model_performance(test_data=test))

    # Print results
    prediction = model.predict(test_data=test_df)
    prediction_df = prediction.as_data_frame().rename(columns={"predict": "TARGET"})
    id_label = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'test_id.csv'))
    submission = pd.concat([id_label, prediction_df], axis=1, sort=False)
    submission.to_csv(os.path.join(para.OUTPUT_PATH, 'submission.csv'), index=False)

    model.plot()
    model._model_json['output']['variable_importances'].\
        as_data_frame().to_csv(os.path.join(para.OUTPUT_PATH, 'R3_VI.csv'))

    plt.rcdefaults()
    # endregion

    print('[{}] Saving model'.format(util.time_now()))
    # model = h2o.load_model('C:\\R3\\DRF_model_python_1530600013847_1')
    model_path = h2o.save_model(model=model, path='/R3', force=True)
    print('The model is saved in:', model_path)

    print('[{}] End'.format(util.time_now()))
