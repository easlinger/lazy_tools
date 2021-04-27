#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:02:52 2020

@author: ena
"""
#%% Imports

## Basics
import warnings
import numpy as np
from autograd import grad, jacobian
import sympy as sym
import scipy as sp
from scipy import signal
import pandas as pd
import functools
import math
import random
import os
import ast
import re

## Pythonic R Paste Command
def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)
def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)

## Autograd
# from __future__ import absolute_import
# from __future__ import print_function
# import matplotlib.pyplot as plt
# import numpy as npo
# import autograd.numpy as np
# from autograd import grad
# from autograd.scipy.integrate import odeint
# from autograd.builtins import tuple
# from autograd.misc.optimizers import adam
# import autograd.numpy.random as npr


## Math
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS

## Visualizations
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages


#%% Pandas

def arraycol_2_multicol(data, col = 0, labels = 0, str2int = 0):
    """
    Convert a dataframe with a single column where each row is an array
    Returns a dataframe where each element of the row arrays is a separate column

    Examples
    --------
    >>> data = pd.DataFrame([[rand(3)], [rand(3)], [rand(3)], [rand(3)]])
    >>> data_new = arraycol_2_multicol(data, labels = ['x1', 'x2', 'x3'])
    >>> data_new
    """
    import re, ast
    data_col = pd.DataFrame(data)
    data_col = pd.DataFrame(data_col.iloc[:, col])
    rows = range(np.shape(data_col)[0])
    columns = range(np.shape(data_col)[1])
    data_col.columns = ['x']
    data_col['x'] = pd.DataFrame([str(data_col.iloc[r, col]) for r in rows]) # arrays -> strings
    data_col['x'] = pd.DataFrame([re.sub(' ', ',', data_col.iloc[r, 0]) for r in rows]) # replace , with space
    data_col['x'] = pd.DataFrame([re.sub(',+', ', ', data_col.iloc[r, 0]) for r in rows]) # remove redundant commas
    data_col['x'] = pd.DataFrame(data_col)['x'].str.strip('[ ').str.strip(' ]') # strip whitespace after/before []
    data_col['x'] = pd.DataFrame(data_col)['x'].str.strip('\n')# strip \n
    # data_col['x'] = pd.DataFrame([re.sub('[[]', '', data_col.iloc[r, col]) for r in rows]) # remove [
    # data_col['x'] = pd.DataFrame([re.sub('[]]', '', data_col.iloc[r, col]) for r in rows]) # remove ]
    data_col['x'] = pd.DataFrame([re.sub('\\n', '', data_col.iloc[r, col]) for r in rows]) # remove \n
    data_col['x'] = data_col['x'].apply(lambda x: re.sub('\s+', '', re.sub('.*nan', 'None', str(x))))
    data_col['x'] = data_col['x'].apply(lambda s: str('[' + s + ']')) # add back in brackets
    data_col['x'] = data_col['x'].apply(ast.literal_eval)

    data_col['x'] = pd.DataFrame([str(data_col.iloc[r, col]) for r in rows]) # arrays -> strings
    data_col['x'] = pd.DataFrame([re.sub(' ', ',', data_col.iloc[r, 0]) for r in rows]) # replace , with space
    data_col['x'] = pd.DataFrame([re.sub(',+', ', ', data_col.iloc[r, 0]) for r in rows]) # remove redundant commas
    data_col['x'] = pd.DataFrame(data_col)['x'].str.strip('[ ').str.strip(' ]') # strip whitespace after/before []
    data_col['x'] = pd.DataFrame(data_col)['x'].str.strip('\n')# strip \n
    # data_col['x'] = pd.DataFrame([re.sub('[[]', '', data_col.iloc[r, col]) for r in rows]) # remove [
    # data_col['x'] = pd.DataFrame([re.sub('[]]', '', data_col.iloc[r, col]) for r in rows]) # remove ]
    data_col['x'] = pd.DataFrame([re.sub('\\n', '', data_col.iloc[r, col]) for r in rows]) # remove \n
    data_col['x'] = data_col['x'].apply(lambda x: re.sub('\s+', '', re.sub('.*nan', 'None', str(x))))

    data_col = data_col['x'].str.split(',', expand=True)

    try:
        if labels == 0: labels = [str('x' + str(i)) for i in columns]
        data_col.columns = labels # name parameter columns
    except:
        print('Unable to reassign labels.')
    return data_col

# tmp = pd.DataFrame(tmpp.reset_index()[tmpp.reset_index().subject == 4]['DV_residualized'].reset_index(drop=True))
# arraycol_2_multicol(tmp)
