# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#Clustering
from sklearn.cluster import KMeans


##################################################
##    PREPROCESSING DATA HELPER FUNCTIONS
##################################################

def create_unknown_dataset(dictionary_df):
    '''
    Creates a dataset containing unknown codes for each column in the azdias dataset

    :param dictionary_df: Dataframe containing feature information of azdias dataset
    :return: df_unknown: Clean dataframe of unknown codes

    '''
    print('Shape of the attribute file: {}'.format(dictionary_df.shape))
    unknown_values = dictionary_df.Meaning.str.contains('unknown', case=False)
    print('Missing values in the array: {}'.format(len(unknown_values[unknown_values.isnull()])))
    unknown_values.fillna(value=False, inplace=True)
    print('After filling NaNs with False: ')
    print(unknown_values.value_counts())
    # Due to the nature of the attribute file, I would need to fix the NaNs above manually
    dictionary_df.loc[821, 'Attribute'] = 'KBA05_AUTOQUOT'
    dictionary_df.loc[2083, 'Attribute'] = 'RELAT_AB'
    df_unknown = dictionary_df[unknown_values]

    return df_unknown

def create_unknown_dictionary(dictionary_df):
    '''
    Creates a dictionary containing unknown codes for each column in the azdias dataset

    :param df_unknown: Clean dataframe of unknown codes
    :return: unknown_dict: Dictionary containing unknown codes
    '''
    df_unknown = create_unknown_dataset(dictionary_df)
    unknown_dict = dict()

    for attribute, unknown_value in zip(df_unknown.Attribute, df_unknown.Value):

        if type(unknown_value) == int:
            values = [unknown_value]
        else:
            values = [int(element) for element in unknown_value.split(',')]

        unknown_dict[attribute] = values

    return unknown_dict
    #print(len(unknown_dict))

def replace_unknown_values(dataset,feature_dict):
    """
    Replaces missing codes with NaNs using a feature dictionary for azdias dataset

    :param  dataset - Azdias dataframe
    :return None
    """

    for column in dataset.columns:
        try:
            dataset[column] = dataset[column].replace(feature_dict[column], np.nan)
        except KeyError:
            pass


def get_missing_and_nas(data, dictionary):
    """
    Obtains the number of missing values and NaNs of every column

     :param data - azdias dataframe
     :param dictionary  - azdias dictionary

     :return col_na - dictionary containing a missing and NaNs counts for every column
    """

    col_na = dict()
    for column in data.columns:
        missing_count = 0
        nas_count = 0
        for elem in data[column]:
            try:
                if elem in dictionary[column]:
                    # print(elem)
                    missing_count += 1
                elif pd.isnull(elem):
                    # print(elem)
                    nas_count += 1
            except KeyError:
                pass
        # print(column, missing_count, nas_count)
        col_na[column] = {nas_count, missing_count}
    return col_na


def remove_nas_rows(dataset):
   '''
   Splits azdias dataset by number of missing values. 25 %

   :param dataset: azdias dataset
   :return:
        df_low: azdias dataset with few missing values
        df_high: azdias dataset with high missing values

   '''
   print("Splitting records with NAs..\n", "Total records: {}".format(dataset.shape[0]))
   rows_NA = dataset.isnull().sum(axis=1)
   # rows_NA.drop(labels=rows_NA[rows_NA[:] <= 0].index, inplace=True)
   n_columns = dataset.shape[1]

   quarter_of_columns = int(n_columns * 0.25)

   rows_NA_low = rows_NA[rows_NA < quarter_of_columns]
   rows_NA_high = rows_NA[rows_NA >= quarter_of_columns]

   print("Records split by {} missing values.\n".format(quarter_of_columns),
          "Shape of resulting dataset: {}\n".format(rows_NA_low.shape),
          "Shape of high NAs dataset: {}\n".format(rows_NA_high.shape))

   df_low = dataset[dataset.index.isin(rows_NA_low.index)]
   df_high = dataset[dataset.index.isin(rows_NA_high.index)]

   return df_low, df_high


def split_cameo(value, wealth):
    '''
    Splits the value of CAMEO_INTL_2015 column by wealth and lifestyle

    :param value: single value of CAMEO_INTL_2015
    :param wealth: flag to return wealth values, otherwise returns lifestyle values
    :return: a number either for wealth or for lifestyle
    '''
    if pd.isnull(value):
        return np.nan

    else:
        value = str(value)

        if wealth:
            # print(value,value[0],value[1])
            return value[0]

        else:
            return value[1]

        # print(value,value[0],value[1])


# type(cameo_values)

def obtain_categorical_columns(df):
    '''
    Obtains a dictionary with the categorical columns grouped by multi and binary
    :param df: azdias dataframe
    :return: cat_col: dictionary containing columns names
    '''
    feature_info = pd.read_csv('feature_info.csv')
    categorical_cols = feature_info.Feature[feature_info.Type == 'Categorical']

    cat_col = dict()
    for col in categorical_cols:
        try:
            if (df[col].nunique() > 2):
                cat_col.setdefault('multi', [])
                cat_col['multi'].append(col)
            else:
                cat_col.setdefault('binary', [])
                cat_col['binary'].append(col)
        except:
            pass

    # updating the cat_col list, removing the CAMEO_DEU_2015 from the dictionary
    cat_col['multi'].remove('CAMEO_DEU_2015')

    return cat_col


def clean_azdias(df, unknown_dict, outlier_columns, cat_col):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data

    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """

    mainstream = (1, 3, 5, 8, 10, 12, 14)
    decade = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60, 8: 70, 9: 70, 10: 80, 11: 80, 12: 80, 13: 80, 14: 90,
              15: 90}
    # convert missing value codes into NaNs, ...
    replace_unknown_values(df, unknown_dict)

    # Removing the outlier columns from the dataset.
    try:
        df.drop(outlier_columns, axis=1, inplace=True)
    except:
        pass

    df, df_high_na = remove_nas_rows(df)

    # Manually replacing missign values for CAMEO_DEU, CAMEO_DEUG and CAMEO_INTL
    df.loc[df['CAMEO_DEU_2015'] == 'XX', 'CAMEO_DEU_2015'] = np.nan
    # df.loc[df['CAMEO_DEUG_2015'] == 'X', 'CAMEO_DEUG_2015'] = np.nan
    # df.loc[:, 'CAMEO_DEUG_2015'] = df.loc[:, 'CAMEO_DEUG_2015'].astype('float')
    # df.loc[df['CAMEO_INTL_2015'] == 'XX', 'CAMEO_INTL_2015'] = np.nan
    # df.loc[:, 'CAMEO_INTL_2015'] = df.loc[:, 'CAMEO_INTL_2015'].astype('float')

    # select, re-encode, and engineer column values.
    df['OST_WEST_KZ'].replace(['W', 'O'], [1, 0], inplace=True)
    df.drop('CAMEO_DEU_2015', axis=1, inplace=True)
    df = pd.get_dummies(df, columns=cat_col['multi'])

    col_move = np.where(df['PRAEGENDE_JUGENDJAHRE'].isin(mainstream), 0,
                        np.where(df['PRAEGENDE_JUGENDJAHRE'].notnull(), 1, np.nan))
    col_dec = df['PRAEGENDE_JUGENDJAHRE'].map(decade)

    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = col_move
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = col_dec

    df['CAMEO_INTL_2015_WEALTH'] = df['CAMEO_INTL_2015'].apply(split_cameo, wealth=1).astype('float')
    df['CAMEO_INTL_2015_LIFESTAGE'] = df['CAMEO_INTL_2015'].apply(split_cameo, wealth=0).astype('float')

    df['EINGEFUEGT_AM_YEAR'] = df['EINGEFUEGT_AM'].str[:4].astype(int)

    # Finally we drop the original columns

    df.drop(['CAMEO_INTL_2015', 'PRAEGENDE_JUGENDJAHRE', 'LP_LEBENSPHASE_FEIN', 'EINGEFUEGT_AM'], axis=1, inplace=True)
    # Deleting the 'ONLINE_PURCHASE', 'CUSTOMER_GROUP', 'PRODUCT_GROUP' columns from customers
    df.drop(columns=['ONLINE_PURCHASE', 'CUSTOMER_GROUP', 'PRODUCT_GROUP'], inplace=True)

    # Return the cleaned dataframe.
    return df, df_high_na


def clean_mailout(df, unknown_dict, outlier_columns, cat_col):
    """
    Perform feature trimming, re-encoding, and engineering for mailout data

    :param  df -   Mailout dataframe
    :return df - Trimmed and cleaned Mailout DataFrame
    """
    mainstream = (1, 3, 5, 8, 10, 12, 14)
    decade = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60, 8: 70, 9: 70, 10: 80, 11: 80, 12: 80, 13: 80, 14: 90,
              15: 90}
    # convert missing value codes into NaNs, ...
    replace_unknown_values(df, unknown_dict)

    # Removing the outlier columns from the dataset.
    try:
        df.drop(outlier_columns, axis=1, inplace=True)
    except:
        pass

    df, df_high_na = remove_nas_rows(df)

    # Manually replacing missing values for CAMEO_DEU, CAMEO_DEUG and CAMEO_INTL
    df.loc[df['CAMEO_DEU_2015'] == 'XX', 'CAMEO_DEU_2015'] = np.nan
    df.loc[df['CAMEO_DEUG_2015'] == 'X', 'CAMEO_DEUG_2015'] = np.nan
    df.loc[:, 'CAMEO_DEUG_2015'] = df.loc[:, 'CAMEO_DEUG_2015'].astype('float')
    df.loc[df['CAMEO_INTL_2015'] == 'XX', 'CAMEO_INTL_2015'] = np.nan
    df.loc[:, 'CAMEO_INTL_2015'] = df.loc[:, 'CAMEO_INTL_2015'].astype('float')

    # select, re-encode, and engineer column values.
    df['OST_WEST_KZ'].replace(['W', 'O'], [1, 0], inplace=True)
    df.drop('CAMEO_DEU_2015', axis=1, inplace=True)
    df = pd.get_dummies(df, columns=cat_col['multi'])

    col_move = np.where(df['PRAEGENDE_JUGENDJAHRE'].isin(mainstream), 0,
                        np.where(df['PRAEGENDE_JUGENDJAHRE'].notnull(), 1, np.nan))
    col_dec = df['PRAEGENDE_JUGENDJAHRE'].map(decade)

    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = col_move
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = col_dec

    df['CAMEO_INTL_2015_WEALTH'] = df['CAMEO_INTL_2015'].apply(split_cameo, wealth=1).astype('float')
    df['CAMEO_INTL_2015_LIFESTAGE'] = df['CAMEO_INTL_2015'].apply(split_cameo, wealth=0).astype('float')

    df['EINGEFUEGT_AM_YEAR'] = df['EINGEFUEGT_AM'].str[:4].astype(float)

    # Finally we drop the original columns

    df.drop(['CAMEO_INTL_2015', 'PRAEGENDE_JUGENDJAHRE', 'LP_LEBENSPHASE_FEIN', 'EINGEFUEGT_AM'], axis=1, inplace=True)

    # Return the cleaned dataframe.
    return df, df_high_na

##################################################
##    UNSUPERVISED LEARNING HELPER FUNCTIONS
##################################################

def do_pca(n_components, data):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    :param n_components: the number of principal components to create
    :param data: the data you would like to transform

    :return:
        pca : the pca object created after fitting the data
        X_pca : the transformed X matrix with new number of components
    '''
    pca = PCA(n_components)
    X_pca = pca.fit_transform(data)
    return pca, X_pca


def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components

    :param pca : the result of instantiation of PCA in scikit learn
    :return None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i] * 100)[:4])), (ind[i] + 0.2, vals[i]), va="bottom", ha="center",
                    fontsize=12)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


def get_component(pca, desired_variance):
    '''
    Returns the number of components needed to obtain a certain variance
    :param pca: the result of instantiation of PCA in scikit learn
    :param desired_variance: desired variance
    :return: component: number of components needed to obtain the desired_variance
    '''
    variance = 0
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    for component in range(0, num_components):
        variance += vals[component]
        # print(num_components, component, variance, vals[component])

        if variance >= desired_variance:
            print("The number of components needed to obtain {} of variance is {}".format(variance, component + 1))
            return component + 1


def plot_component(full_dataset, pca, n_components, color='purple'):
    '''
    Maps weights for the n_components to corresponding feature names and plots the component with its features names
    Sorted by weights
    :param full_dataset: azdias dataset
    :param pca: the result of instantiation of PCA in scikit learn
    :param n_components: number of component to map
    :param color: color of the bars
    :return: A Plot with the weights for the n_component
    '''
    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions
    components = components.iloc[n_components-1]

    comp_pos =  components.sort_values(ascending=False)[:10]
    comp_neg = components.sort_values(ascending=True)[:10]
    comp_merged = pd.concat([comp_pos, comp_neg])
    #comp_merged

    comp_merged.plot(kind='bar', color=color, title=dimensions[n_components-1])

#comp_end = components.sort_values(ascending=True)[:10]