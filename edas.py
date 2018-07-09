# --------------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------------
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
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
prev_df = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'previous_application.csv'))




bureau_df = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'bureau.csv'))

bureau_bal_df = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'bureau_balance.csv'))

cc_balance_df = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'credit_card_balance.csv'))

installment_df = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'installments_payments.csv'))

cash_df = pd.read_csv(os.path.join(para.RESOURCE_PATH, 'POS_CASH_balance.csv'))


