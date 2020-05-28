#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following code is to examine a particular case of missing data: missingness
at random. In this case, the independent variable (PPP adjusted per capita
GDP) has missing values and their missingness is correlated with the 
corresponding dependent variable (Real per capita GDP growth) values: countries
growing at a higher rate are more likely to have missing data.  
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

def simulate_na(df, fraction = 0.5, N = 100):
    """
    Function to simulate missing values.
    """
    df_list = []
    for i in range(N):
        df_cp = df.copy()
        df_cp.loc[df_cp.sample(frac = fraction, replace = False, weights = 'weight').index, 'pc_GDP_PPP'] = np.nan
        df_list.append(df_cp)
    return df_list

def run_ols(df_list):
    """
    This function runs ols on a list of dataframes and returns the results
    as a dictionary of lists.
    """
    ols_rslts_dict = {'y_on_x': [], 'x_on_y': []}
    for df in df_list:
        model = sm.OLS.from_formula('pc_GDP_growth ~ pc_GDP_PPP', df, missing = 'drop')
        ols_rslts_dict['y_on_x'].append(model.fit())
        model = sm.OLS.from_formula('pc_GDP_PPP ~ pc_GDP_growth', df, missing = 'drop')
        ols_rslts_dict['x_on_y'].append(model.fit())
    return ols_rslts_dict

def get_betas(ols_rslts_dict, scope):
    """
    This function gets betas from a dictionary of ols results and returns a 
    dataframe.
    """
    param_list = []
    for result in ols_rslts_dict['x_on_y']:
        item = {}
        item['scope'] = scope
        item['type'] = 'x_on_y' 
        item['beta0'] = result.params['Intercept']
        item['beta1'] = result.params['pc_GDP_growth']
        item['std0'] = result.bse['Intercept']
        item['std1'] = result.bse['pc_GDP_growth']
        param_list.append(item)
    for result in ols_rslts_dict['y_on_x']:
        item = {}
        item['scope'] = scope
        item['type'] = 'y_on_x' 
        item['beta0'] = result.params['Intercept']
        item['beta1'] = result.params['pc_GDP_PPP']
        item['std0'] = result.bse['Intercept']
        item['std1'] = result.bse['pc_GDP_PPP']
        param_list.append(item)
    return pd.DataFrame(param_list)

def impute_na(df_list, betas_df):
    """
    This function imputes the missing values using a random regression
    approach.
    """
    imputed_df_list = []
    for df, i in zip(df_list, betas_df.index):
        imputed_df = df.copy()
        beta0 = betas_df.loc[i, 'beta0']
        beta1 = betas_df.loc[i, 'beta1']
        std0 = betas_df.loc[i, 'std0']
        std1 = betas_df.loc[i, 'std1']
        for i in imputed_df.loc[imputed_df.pc_GDP_PPP.isna()].index:
            imputed_df.pc_GDP_PPP[i] = np.random.normal(loc = beta0, scale = std0, size = 1) + \
                                       (np.random.normal(loc = beta1, scale = std1, size = 1) * \
                                        imputed_df.pc_GDP_growth[i])
        imputed_df_list.append(imputed_df)
    return imputed_df_list

def summarize(betas_df):
    """
    This function summarizes the parameters to plot.
    """
    summary_df = betas_df.loc[betas_df.type == 'y_on_x'].groupby('scope').mean()[['beta1', 'std1']].reset_index()
    new_rows = []
    for i in summary_df.index:
        row1 = {'scope': '', 'beta1': np.nan, 'std1': np.nan}
        row2 = {'scope': '', 'beta1': np.nan, 'std1': np.nan}
        row1['scope'] = summary_df.scope[i]
        row2['scope'] = summary_df.scope[i]
        row1['beta1'] = summary_df.beta1[i] + 1.96 * summary_df.std1[i]
        row2['beta1'] = summary_df.beta1[i] - 1.96 * summary_df.std1[i]
        new_rows.append(row1)
        new_rows.append(row2)
    summary_df = summary_df.append(new_rows)
    return summary_df
