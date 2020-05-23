#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following code is to examine a particular case of missing data: missingness
at random. In this case, the independent variable (PPP adjusted per capita
GDP) has missing values and their missingness is correlated with the 
corresponding dependent variable (Real per capita GDP growth) values: countries
growing at a higher rate are more likely to have missing data.  
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# =============================================================================
# Function to flatten the WB series and convert them to a dataframe.
# The dataframe will not include aggregates but only countries 
# =============================================================================
def flattened_df(nested_list, country_iso3_code):
    df_list = []
    for country in nested_list:
        value = np.nan
        if country['value'] != None:
            value = float(country['value'])
        flattened_dict = {'country': country['country'], \
                          'country_iso3_code': country['countryiso3code'], \
                          'date': int(country['date']), \
                          'value': value}
        df_list.append(pd.DataFrame(flattened_dict))
    df = pd.concat(df_list).loc['value', :].reset_index(drop = True)
    df = df.merge(right = country_iso3_code, \
                  how = 'inner', \
                  left_on = 'country_iso3_code', \
                  right_on = 'ISO-alpha3 code')\
           .sort_values(['country', 'date'])\
           .loc[:, ['country', 'country_iso3_code', 'date', 'value']]
    return df

# =============================================================================
# Function to create missing values.
# It takes weight to correlate missingness with existing data. 
# =============================================================================
def create_na(data_frame, fraction = 0.5):
    data_frame_cp = data_frame.copy()
    data_frame_cp.loc[data_frame.sample(frac = fraction, \
                                    replace = False, \
                                    weights = 'weights').index, 'per_capita_GDP'] = np.nan 
    return data_frame_cp 

# =============================================================================
# Function to simulate missing values.
# By default, N = 100. 
# =============================================================================
def simulate_na(data_frame, x, fraction = 0.5, N = 100):
    data_frame_list = []
    for i in range(N):
        data_frame_list.append(create_na(data_frame, x, fraction = 0.5))
    return data_frame_list

# =============================================================================
# Function to simulate missing values.
# By default, N = 100. 
# =============================================================================
def fill_na(data_frame, beta):
    for i in data_frame.loc[data_frame.per_capita_GDP.isna()].index:
        data_frame.per_capita_GDP[i] = np.random.normal(loc = beta['beta']['intercept'], \
                                                        scale = beta['std']['intercept'], \
                                                        size = 1) + \
                                       np.random.normal(loc = beta['beta']['per_capita_GDP_growth'], \
                                                        scale = beta['std']['per_capita_GDP_growth'], \
                                                        size = 1) * \
                                       data_frame.per_capita_GDP_growth[i]
    return data_frame

# =============================================================================
# Function to simulate missing values.
# By default, N = 100. 
# =============================================================================
def impute_na(data_frame_list, beta_list):
    data_frame_list_cp = []
    for data_frame, beta in zip(data_frame_list, beta_list):
        data_frame_cp = data_frame.copy()
        data_frame_cp = fill_na(data_frame_cp, beta)
        data_frame_list_cp.append(data_frame_cp)
    return data_frame_list_cp

def run_ols(data_frame_list):
    ols_results_dict = {'y_on_x': [], 'x_on_y': []}
    for data_frame in data_frame_list:
        model = sm.OLS.from_formula('per_capita_GDP_growth ~ per_capita_GDP', data_frame, missing = 'drop')
        ols_results_dict['y_on_x'].append(model.fit())
        model = sm.OLS.from_formula('per_capita_GDP ~ per_capita_GDP_growth', data_frame, missing = 'drop')
        ols_results_dict['x_on_y'].append(model.fit())
    return ols_results_dict

def create_beta_dict(ols_results_dict):
    beta_dict = {'x_on_y': [], 'y_on_x': []}
    beta = {}
    std = {}
    beta_list = []
    for result in ols_results_dict['x_on_y']:
        beta['intercept'] = result.params['Intercept']
        beta['per_capita_GDP_growth'] = result.params['per_capita_GDP_growth']
        std['intercept'] = result.bse['Intercept']
        std['per_capita_GDP_growth'] = result.bse['per_capita_GDP_growth']
        beta_list.append({'beta': beta, 'std': std})
    beta_dict['x_on_y'] = beta_list
    beta = {}
    std = {}
    beta_list = []
    for result in ols_results_dict['y_on_x']:
        beta['intercept'] = result.params['Intercept']
        beta['per_capita_GDP'] = result.params['per_capita_GDP']
        std['intercept'] = result.bse['Intercept']
        std['per_capita_GDP'] = result.bse['per_capita_GDP']
        beta_list.append({'beta': beta, 'std': std})
    beta_dict['y_on_x'] = beta_list
    return beta_dict

# =============================================================================
#  Function to extract data through the WB API and UN web page.
#  Returns a dataframe.
# =============================================================================
def etl():
    url = 'http://api.worldbank.org/v2/country/all/indicator'
    indicator_list = ['NY.GDP.PCAP.KD.ZG', \
                      'NY.GDP.PCAP.PP.CD']
    parameters = {'date': '1989:2018', \
                  'footnote': 'n', \
                  'format': 'json', \
                  'per_page': 7920}
    results_list = []
    for indicator in indicator_list:
        results = requests.get(f'{url}/{indicator}', \
                               params = parameters)
        if results.status_code == 200:
            results_list.append(results.json())
        else:
            print('Failed request')
    country_iso3_code = pd.read_html('https://unstats.un.org/unsd/methodology/m49/')
    country_iso3_code = country_iso3_code[0]  
    list_per_capita_GDP_growth = results_list[0][1] 
    list_per_capita_GDP_PPP = results_list[1][1]
    df_per_capita_GDP_growth = flattened_df(list_per_capita_GDP_growth, country_iso3_code)
    df_per_capita_GDP_PPP = flattened_df(list_per_capita_GDP_PPP, country_iso3_code)
    # Drop missing values, and prepare the initial values as well as mean growth rates
    df_per_capita_GDP_PPP = df_per_capita_GDP_PPP.loc[(df_per_capita_GDP_PPP.date >= 1994) & \
                                                      (df_per_capita_GDP_PPP.date <= 1996) & \
                                                      (~df_per_capita_GDP_PPP.value.isna())]\
                                                 .groupby('country').value.mean()
    df_per_capita_GDP_growth = df_per_capita_GDP_growth.loc[(df_per_capita_GDP_growth.date >= 1995) & \
                                                            (df_per_capita_GDP_growth.date <= 2018) & \
                                                            (~df_per_capita_GDP_growth.value.isna())]
    df_per_capita_GDP_growth['value'] = (df_per_capita_GDP_growth.value + 100) / 100
    df_per_capita_GDP_growth = pd.concat([df_per_capita_GDP_growth.groupby('country').value.prod(), \
                                          df_per_capita_GDP_growth.groupby('country').value.count()], \
                                         axis = 1, \
                                         ignore_index = True)
    # Compute geometric mean
    df_per_capita_GDP_growth['per_capita_GDP_growth'] = (df_per_capita_GDP_growth.iloc[:, 0] ** \
                                                         (1/df_per_capita_GDP_growth.iloc[:, 1]) - 1) * 100     
    df_per_capita_GDP_growth = df_per_capita_GDP_growth[['per_capita_GDP_growth']]                                            
    df_convergence = pd.merge(left = df_per_capita_GDP_PPP, 
                              right = df_per_capita_GDP_growth, \
                              how = 'inner', \
                              on = 'country')
    df_convergence['per_capita_GDP'] = np.log(df_convergence.value)
    df_convergence = df_convergence.drop(columns = 'value')
    return df_convergence

# =============================================================================
# Function to create weights. 
# 
# =============================================================================
def create_weights(data_frame):
    data_frame['weights'] = 0
    data_frame.loc[data_frame.per_capita_GDP_growth > 0, \
                   'weights'] = data_frame.loc[data_frame.per_capita_GDP_growth > 0, \
                                               'per_capita_GDP_growth'] / \
                                data_frame.per_capita_GDP_growth.sum()                                
    return data_frame 

def create_beta_df(betas_dict):
    betas_list = [{'mean_beta': np.mean([item['beta']['per_capita_GDP'] \
                                         for item in betas_dict['y_on_x']]), \
                   'mean_std': np.mean([item['std']['per_capita_GDP'] \
                                         for item in betas_dict['y_on_x']]),
                   'scope': 'complete case',
                   'type': 'y_on_x'},
                  {'mean_beta': np.mean([item['beta']['per_capita_GDP_growth'] \
                                         for item in betas_dict['x_on_y']]), \
                   'mean_std': np.mean([item['std']['per_capita_GDP_growth'] \
                                         for item in betas_dict['x_on_y']]),
                   'scope': 'complete case',
                   'type': 'x_on_y'}]
    model_full_sample_y_on_x = sm.OLS.from_formula('per_capita_GDP_growth ~ per_capita_GDP', data = df_convergence)
    results_full_sample_y_on_x = model_full_sample_y_on_x.fit()
    betas_list.append({'mean_beta': results_full_sample_y_on_x.params['per_capita_GDP'], \
                       'mean_std': results_full_sample_y_on_x.bse['per_capita_GDP'],
                       'scope': 'full sample',
                       'type': 'y_on_x'})
    model_full_sample_x_on_y = sm.OLS.from_formula('per_capita_GDP ~ per_capita_GDP_growth', data = df_convergence)
    results_full_sample_x_on_y = model_full_sample_x_on_y.fit()
    betas_list.append({'mean_beta': results_full_sample_x_on_y.params['per_capita_GDP_growth'], \
                       'mean_std': results_full_sample_x_on_y.bse['per_capita_GDP_growth'],
                       'scope': 'full sample',
                       'type': 'x_on_y'})
    df_betas = pd.DataFrame(betas_list)
    df_betas = df_betas.set_index(['scope', 'type'])
    df_betas['upper_tick'] = df_betas.mean_beta + 1.96 * df_betas.mean_std
    df_betas['lower_tick'] = df_betas.mean_beta - 1.96 * df_betas.mean_std
    df_betas = df_betas.drop(columns = 'mean_std').stack()
    for i in range(2):
        df_betas = df_betas.reset_index(level = i)
    df_betas = df_betas.rename(columns = {df_betas.columns[2]: 'beta'})
    df_betas = df_betas.drop(columns = 'level_1')
    return df_betas

# =============================================================================
# The main body of the code.
# =============================================================================
df_convergence = etl()
df_convergence = create_weights(df_convergence)
np.random.seed(1234)
df_convergence_na = create_na(df_convergence, 'per_capita_GDP')

ax = sns.regplot(df_convergence.per_capita_GDP, df_convergence.per_capita_GDP_growth, fit_reg = False, color = 'r')
ax = sns.regplot(df_convergence_na.per_capita_GDP, df_convergence_na.per_capita_GDP_growth, ci = None)
ax.set_xlabel('PPP adjusted per capita GDP in 1995')
ax.set_ylabel('Annual growth in per capita GDP\n1995-2018')
ax.set_title('Income Convergence, 1995-2018 (Complete Case)')
ax.legend(['complete-case regression', 'set to missing'])
plt.show()

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


