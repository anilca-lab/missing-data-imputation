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

def flatten_wb_api_response(response_list):
    """
    Function to flatten the WB series and convert to a dataframe.
    """
    flattened_list = []
    for response in response_list:
        data_series = response[1]
        for country in data_series:
            value = np.nan
            if country['value'] != None:
                value = country['value']
            data_dict = {'indicator' : country['indicator']['id'], 
                         'country' : country['country']['value'],
                         'country_iso3_code' : country['countryiso3code'],
                         'year' : int(country['date']),
                         'value' : float(value)}
            flattened_list.append(data_dict)
    wb_api_df = pd.DataFrame(flattened_list)
    return wb_api_df

def clean_data(df, start = 1995, stop = 2018):
    """
    Function to drop aggregates, unstack by indicators, and countries 
    with missing values so that we can generate a carefully designed set of 
    missing values based on a specific mechanism.
    """
    country_iso3_code = pd.read_html('https://unstats.un.org/unsd/methodology/m49/')
    country_iso3_code = country_iso3_code[0]['ISO-alpha3 code']
    df = df.loc[df.country_iso3_code.isin(country_iso3_code)]
    df = df.set_index(['indicator', 'country_iso3_code', 'country', 'year']).unstack(level = 0)
    df.columns = df.columns.get_level_values(1)
    df = df.rename(columns = {'NY.GDP.PCAP.KD.ZG': 'pc_GDP_growth',
                              'NY.GDP.PCAP.PP.CD': 'pc_GDP_PPP'})
    df = df.reset_index()
    df = df.loc[(df.year >= (start - 1)) & (df.year <= stop)]
    df = df.dropna()
    return df