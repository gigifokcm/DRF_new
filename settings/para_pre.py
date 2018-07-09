import os

PARA_FUNCTION = "DATAPROCESSING"
PRE_VERSION = ''
VERSION = 'R2'

DEBUG = False

# PATH
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_INPUT_PATH = os.path.join(BASE_PATH, 'resources/' + str("" if PRE_VERSION == "" else (PRE_VERSION + "/")))
INPUT_PATH = os.path.join(BASE_PATH, ('resources/' + VERSION + "/"))

# DATA
TARGET_LABEL = ['TARGET']
ID_LABEL = ['SK_ID_CURR']
TRAIN_LABEL = ""
PREDICTOR_ENCODING = ""
TARGET_ENCODING = ""

# Adjustments
MINORITY_DATA_TOL = 0.01
# for object data or integer, the program will consider the distribution of the predictions....

# SETTING
RUN_DATACLEAN = 1
RUN_TARGET_ANA = 1
RUN_MISSING_DATA = 1
RUN_PREDICTOR_ANA = 1
RUN_PRINTCLEANEDRESULTS = 0
RUN_ENCODING = 0
RUN_IMPUTATION = 0
RUN_NORMALISATION = 0


