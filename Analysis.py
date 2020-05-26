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


"""
# =============================================================================
# The main body of the code.
# =============================================================================



df_convergence_na_list = simulate_na(df_convergence, 'per_capita_GDP')
convergence_na_ols_dict = run_ols(df_convergence_na_list)
betas_dict = create_beta_dict(convergence_na_ols_dict)
df_betas = create_beta_df(betas_dict)
ax = sns.pointplot(x = 'scope', y = 'beta', data = df_betas.loc[df_betas.index == 'y_on_x'], \
                   order = ['full sample', 'complete case'], capsize = 0.1)
ax.set_xlabel('')
ax.set_ylabel('Coefficient for per capita GDP\n(Ticks indicate 95% confidence interval)')
ax.set_title('Income Convergence, 1995-2018\nSimulation Results with N = 100')
plt.show()
ax = sns.pointplot(x = 'scope', y = 'beta', data = df_betas.loc[df_betas.index == 'x_on_y'], \
                   order = ['full sample', 'complete case'], capsize = 0.1)
ax.set_xlabel('')
ax.set_ylabel('Coefficient for per capita GDP growth\n(Ticks indicate 95% confidence interval)')
ax.set_title('Regression of Initial Income Level on the Income Growth\nSimulation Results with N = 100')
plt.show() 

# Imputation

df_convergence_na_imputed_list = impute_na(df_convergence_na_list, betas_dict['x_on_y'])
ax = sns.regplot(df_convergence.per_capita_GDP, df_convergence.per_capita_GDP_growth, fit_reg = False, color = 'r')
ax = sns.regplot(df_convergence_na_imputed_list[0].per_capita_GDP, df_convergence_na_imputed_list[0].per_capita_GDP_growth, ci = None)
ax.set_xlabel('PPP adjusted per capita GDP in 1995')
ax.set_ylabel('Annual growth in per capita GDP\n1995-2018')
ax.set_title('Income Convergence, 1995-2018 (Imputed Values)')
ax.legend(['imputed-values regression', 'set to missing'])
plt.show()

convergence_na_imputed_ols_dict = run_ols(df_convergence_na_imputed_list)
betas_imputed_dict = create_beta_dict(convergence_na_imputed_ols_dict)
df_betas_imputed = create_beta_df(betas_imputed_dict)
df_betas_imputed.loc[df_betas_imputed.scope == 'complete case', 'scope'] = 'imputed values'
ax = sns.pointplot(x = 'scope', y = 'beta', data = df_betas_imputed.loc[df_betas_imputed.index == 'y_on_x'], \
                   order = ['full sample', 'imputed values'], capsize = 0.1)
ax.set_xlabel('')
ax.set_ylabel('Coefficient for per capita GDP\n(Ticks are to indicate standard deviation)')
ax.set_title('Regression of Income Growth on the Initial Income Level')
plt.show()




"""

