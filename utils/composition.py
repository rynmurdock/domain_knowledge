# -*- coding: utf-8 -*-
"""
**Code adopted from pymatgen library
"""

# set path to element_properties folder
path = r'data/element_properties/'


import collections
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot

# **
def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.\d]*)", f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError("{} is an invalid formula!".format(f))
    return sym_dict

# **
def parse_formula(formula):
    """
    Args:
        formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3
    Returns:
        Composition with that formula.
    Notes:
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace("@", "")

    m = re.search(r"\(([^\(\)]+)\)\s*([\.\d]*)", formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    return get_sym_dict(formula, 1)

# **
class CompositionError(Exception):
    """Exception class for composition errors"""
    pass

# **
def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {}
    for key in elamt:
        comp_frac[key] = elamt[key] / natoms
    return comp_frac

# **
def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    return elamt

def _assign_features(formula, elem_props):
    try:
        fractional_composition = _fractional_composition(formula)
        element_composition = _element_composition(formula)
        avg_feature = np.zeros(len(elem_props.iloc[0]))
        sum_feature = np.zeros(len(elem_props.iloc[0]))
        for key in fractional_composition:
            try:
                avg_feature += elem_props.loc[key].values * fractional_composition[key]
                sum_feature += elem_props.loc[key].values * element_composition[key]
            except:
               # print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                return np.array([np.nan]*len(elem_props.iloc[0])*4)
        var_feature = elem_props.loc[list(fractional_composition.keys())].var()
        range_feature = elem_props.loc[list(fractional_composition.keys())].max()-elem_props.loc[list(fractional_composition.keys())].min()

#        features = pd.DataFrame(np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)]))
        features = np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)])
        return features.transpose()
    except:
#        print('There was an error with the formula: "'+ formula + '", please check the formatting')
        return np.array([np.nan]*len(elem_props.iloc[0])*4)

def generate_features(df, features_style='jarvis', reset_index=True):
    '''
    Parameters
    ----------
    df: Pandas.DataFrame()
        Two column dataframe of form: 
            df.columns.values = array(['formula', 'target'], dtype=object)

    Return
    ----------
    X: pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value 
        for  the dataset
    y: pd.Series()
        Target values
    '''

    if features_style is not None:
        elem_props = pd.read_csv(path + features_style + '.csv')
        elem_props.index = elem_props['element'].values
        elem_props.drop(['element'], inplace=True, axis=1)

    column_names = np.concatenate(['avg_'+elem_props.columns.values,
                                   'sum_'+elem_props.columns.values,
                                   'var_'+elem_props.columns.values,
                                   'range_'+elem_props.columns.values])

    # make empty list where we will store the feature vectors
    features = []
    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop
    for formula, target in zip(df['formula'], df['target']):
        pass
        features.append(_assign_features(formula, elem_props))
        targets.append(target)
        formulae.append(formula)

    # split feature vectors and targets as X and y
    X = pd.DataFrame(features, columns=column_names, index=df.index.values)
    y = pd.Series(targets, index=df.index.values, name='target')
    formulae = pd.Series(formulae, index=df.index.values, name='formula')
    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of Nan values.
    X.dropna(inplace=True, how='all')
    y = y.loc[X.index]
    formulae = formulae.loc[X.index]

    if reset_index is True:
        # reset dataframe indices to simplify code later.
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        formulae.reset_index(drop=True, inplace=True)

    # get the column names
    cols = X.columns.values
    # find the mean value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the columns mean value
    X[cols] = X[cols].fillna(median_values.iloc[0])
    return X, y, formulae
