# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:18:02 2021

@author: Christoph Frey
"""

# Libraries
import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import statsmodels as sm

# Data
T = 1000
N = 7
returns = pd.DataFrame(np.random.randn(T,N))

# Rolling regression
WINDOW = 100
y = returns.iloc[:, 0]
X = returns.iloc[:, 1:]

# Restrictions
L = np.array(([0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1]))
r = [1, 1]


def RollingRLS(y, X, constant_indicator, window, L, r):
    '''
    RollingRLS: Calculate rolling restricted least squares
    
    Requirments:  statsmodels v0.13.0rc0 with RollingOLS see
    https://www.statsmodels.org/stable/generated/statsmodels.regression.rolling.RollingOLS.html
    
    Input:
    ==========
    y : (1 x N) dependent variable
    X : (T x N) regressor matrix
    constant_indicator: bool to indicate to include constant in regression
    window: scalar, rolling window length
    L: (M x N) restriction matrix for m restrictions
    r: (M x 1) restriction vector
    
    Outout:
    ==========
    model: statsmodels model object
    '''
    if constant_indicator:
        X = sm.tools.tools.add_constant(X)
    model = RollingOLS(y, X, window=window, expanding=False, min_nobs=X.shape[1]).fit(params_only=True)
    model_restricted = model
    for i in enumerate(X.index):
        tmp_X = X.iloc[max(0,i[0]-WINDOW):i[0],:]
        try:
            tmp_XtX_inv = np.linalg.inv(np.dot(np.transpose(tmp_X), tmp_X))
            RA_1 = np.dot(tmp_XtX_inv, np.transpose(L))
            RA_2 = np.linalg.inv(np.linalg.multi_dot([L, tmp_XtX_inv, np.transpose(L)]))
            tmp_beta_ols = model.params.loc[i[1],:]
            RA_3 = np.dot(L, tmp_beta_ols) - np.array(r)
            RA = RA_1.dot(RA_2).dot(RA_3)
            tmp_beta_rls = tmp_beta_ols - RA
            model_restricted.params.loc[i[1],:] = tmp_beta_rls
        except:
            pass
    return model_restricted


model_restricted = RollingRLS(y, X, True, WINDOW, L, r)
betas_restricted = model_restricted.params