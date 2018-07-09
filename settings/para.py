import os

PARA_FUNCTION = "main"
INPUT_VERSION = ''
INPUT_SUFFIX = '_imputed'
VERSION = 'R2'

# PATH
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_PATH = os.path.join(BASE_PATH, ('resources/' + INPUT_VERSION + "/"))
OUTPUT_PATH = os.path.join(BASE_PATH, ('output/' + VERSION + "/"))
MODEL_PATH = os.path.join((str(RESOURCE_PATH) + VERSION + "/"), 'pred_model.h3')

# DATA
TRAIN_LABEL = ['EXT_SOURCE_3', 'EXT_SOURCE_2','EXT_SOURCE_1']
TARGET_LABEL = ['TARGET']
SUBMISSION_LABEL = 'TARGET'

# RUN
RUN_PCA = 1
TEST_SIZE = 0.1
NN_PARA = {'EMBEDDING_DIM': 300,
           'LSTM_UNITS': 64,
           'DENSE_UNITS': 128,
           'DROPOUT_RATE': 0.2,
           'EPOCHS': 10,
           'TRAIN_BATCH_SIZE': 256,
           'PRED_BATCH_SIZE': 256
           }
