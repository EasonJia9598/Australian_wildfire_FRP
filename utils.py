
# import all related libraries
import pandas as pd
import warnings
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import joblib
import datetime
import math
from sklearn.base import BaseEstimator, TransformerMixin

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# get basemap for geographical plot
from mpl_toolkits.basemap import Basemap

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# for matplotlib ploting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor




###############################################
def max_print_out(pattern=False):
    '''It will maximize print out line and set float format with .2f'''
    number = None if pattern else 10
    # Set options to avoid truncation when displaying a dataframe
    pd.set_option("display.max_rows", number)
    pd.set_option("display.max_columns", 20)
    # Set floating point numbers to be displayed with 2 decimal places
    pd.set_option('display.float_format', '{:.2f}'.format)
    # for showing all entities 




################################  FUNCTION  ################################

    
#---------------------------Load Data---------------------------------------

def loading_data(path):
    '''This function will load mutiple data set and combine them together'''
    if len(path) > 1:
        data = pd.read_csv(path[0])
        for i in range(len(path)-1):
            data = pd.concat([data,pd.read_csv(path[i+1])] , axis=0) 
        # reset index for resolving duplicate indexing probelem
        data = data.reset_index(drop=False)
        # there is a column called index. We won't use it
        # Becasue we will set up our own index
        data = data.drop(['index'],axis=1)
        print("Successfully combined ",len(path), " dataset")
        return data
    else:
        data = data.drop(['index'],axis=1)
        print("Successfully load ",len(path), " dataset")
        return pd.read_csv(path[0])
    
#--------------------------Describe columns----------------------------------------

def describe_columns(data, features_name=[]):
    '''This function will help u print out features value counts'''
    if len(features_name) > 1:
        for i in range(len(features_name)):
            print("----------", data[features_name[i]].name,"---------")
            print(data[features_name[i]].value_counts())
    else:
        print("----------", data[features_name[0]].name,"---------")
        print(data[features_name[0]].value_counts())
        

#-------------Function from tutorial 2-----------------------------

def build_continuous_features_report(data_df):
    
    """Build tabular report for continuous features"""

    stats = {
        "Count": len,
        "Miss %": lambda df: df.isna().sum() / len(df) * 100,
        "Card.": lambda df: df.nunique(),
        "Min": lambda df: df.min(),
        "1st Qrt.": lambda df: df.quantile(0.25),
        "Mean": lambda df: df.mean(),
        "Median": lambda df: df.median(),
        "3rd Qrt": lambda df: df.quantile(0.75),
        "Max": lambda df: df.max(),
        "Std. Dev.": lambda df: df.std(),
    }

    contin_feat_names = data_df.select_dtypes("number").columns
    continuous_data_df = data_df[contin_feat_names]

    report_df = pd.DataFrame(index=contin_feat_names, columns=stats.keys())

    for stat_name, fn in stats.items():
        # NOTE: ignore warnings for empty features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            report_df[stat_name] = fn(continuous_data_df)

    return report_df
    
#-------------Function from tutorial 2---------------------------

def build_categorical_features_report(data_df):

    """Build tabular report for categorical features"""

    def _mode(df):
        return df.apply(lambda ft: ft.mode().to_list()).T

    def _mode_freq(df):
        return df.apply(lambda ft: ft.value_counts()[ft.mode()].sum())

    def _second_mode(df):
        return df.apply(lambda ft: ft[~ft.isin(ft.mode())].mode().to_list())

    def _second_mode_freq(df):
        return df.apply(
            lambda ft: ft[~ft.isin(ft.mode())]
            .value_counts()[ft[~ft.isin(ft.mode())].mode()]
            .sum()
        )

    stats = {
        "Count": len,
        "Miss %": lambda df: df.isna().sum() / len(df) * 100,
        "Card.": lambda df: df.nunique(),
        "Mode": _mode,
        "Mode Freq": _mode_freq,
        "Mode %": lambda df: _mode_freq(df) / len(df) * 100,
        "2nd Mode": _second_mode,
        "2nd Mode Freq": _second_mode_freq,
        "2nd Mode %": lambda df: _second_mode_freq(df) / len(df) * 100,
    }

    cat_feat_names = data_df.select_dtypes(exclude="number").columns
    continuous_data_df = data_df[cat_feat_names]

    report_df = pd.DataFrame(index=cat_feat_names, columns=stats.keys())

    for stat_name, fn in stats.items():
        # NOTE: ignore warnings for empty features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            report_df[stat_name] = fn(continuous_data_df)

    return report_df

#-------------One function to plot multiple kinds of graph ---------------------------

# All codes were written by myself

# set keyword only parameter. Since our data set is too big. must be indentified before ploting

def mulitple_function_plots(tight_layout = True, h_space = 0.4,w_space=0.3, fig_size = (10,15) , 
                            *,data,plot_type="histogram",data_type="number"):
    
    '''Plot all features from the dataset, you must specified your dataset by, data = '''
    if data_type == "number":
        feat_names = data.select_dtypes("number").columns 
    elif data_type == "categorical":
        feat_names = data.select_dtypes(exclude="number").columns
        
    # seperate those features into 2 columns
    rows_number = math.ceil(len(feat_names)/2.0)
    
    print("Those features will be plotted in ", rows_number, " rows and 2 columns")
    # print continuous features name
    print(feat_names)
    
    #initialize figure
    fig, axs = plt.subplots(rows_number, 2, figsize=fig_size)
    index = 0
    start = datetime.datetime.now()
    
    #print
    for i in range(rows_number):
        for j in range(2):
            if index < len(feat_names):
                if plot_type == 'histogram': # shortcut for histogram plot
                    sns.histplot(data=data, x=feat_names[index], bins = 30,kde=True, ax=axs[i][j])
                elif plot_type == 'boxplot': # boxplot
                    data.boxplot(column=feat_names[index],ax=axs[i][j], vert=False)
                elif plot_type == 'barplot': # barplot
                    data[feat_names[index]].value_counts().plot.bar(ax=axs[i][j]);
                # set corresponded name of selected features
                axs[i][j].set_xlabel(feat_names[index])
                # end of calculating the time
                end = datetime.datetime.now()
                # print info
                print(index+1, ". Finish Rendering :", feat_names[index],", used",  
                      (end - start).microseconds/1000, "millseconds")
                index += 1
            else:
                break
    #adjust pictures
    plt.subplots_adjust(hspace = h_space,wspace=w_space)
    # add figure title
    fig.suptitle(str(plot_type.title() + " of all " + data_type.title() + " features"), fontweight ="bold")
    # set whether we want to plot a tight_layout figure
    if tight_layout:
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
    return axs


#------------- draw heatmap -----------------------------------------------------

def heatmap_draw(data):
    # Correlation between different variables
    corr = data.corr()
    # Set up the matplotlib plot configuration
    f, ax = plt.subplots(figsize=(12, 10))
    # Configure a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap
    sns.heatmap(corr, annot=True, cmap=cmap)
    plt.title("Heatmap correlation among all features")
    
    
def create_trainning_data(data_df):
  X = data_df.drop("frp", axis=1) # drop labels for training set
  y = data_df["frp"].copy()
  X = number_pipeline.fit_transform(X)
  y = np.array(y.values)
  return X,y


from numpy.ma.core import negative


# #-------------  detect and delete outliers -----------------------------------------------------

# def find_outliers(data_df, parameter,* , drop=False, set_threshold=True, threshold_value = 1550): # deal with outliers

def find_outliers(data_df, parameter,* , drop=False, set_threshold=True, threshold_value = 350): # deal with outliers    
    '''detect and delete outliers '''
    Q1 = data_df[parameter].quantile(0.25)
    Q3 = data_df[parameter].quantile(0.75)
    IQR = Q3-Q1
    
    print(f"IQR = {Q3} - {Q1} = {IQR}")
    print(f"MAX = {(Q3 + 1.5 * IQR)}")
    
    if Q1 > 1.5*IQR :
        print("Min: ", (Q1 - 1.5 * IQR))
    else:
        print("Min is 0")
    
    cut_out_value =  (Q3 + 1.5 * IQR)
    if set_threshold == True:
        cut_out_value = threshold_value
        
    min_outliers_df = data_df[(data_df[parameter] < (Q1 - 1.5 * IQR))]
    max_outliers_df = data_df[(data_df[parameter] > cut_out_value)]  
    negative_outliers_df = data_df[(data_df[parameter] <= 0)]         
    print("Num of min outliers: ", len(min_outliers_df))
    print("Num of max outliers: ", len(max_outliers_df))
    print("Num of negative outliers: ", len(negative_outliers_df))
    print("Num of the original data set's whole instance", len(data_df))
    print("Rate of purged data/total data", len(max_outliers_df)/ len(data_df))
    if drop:
        return data_df.drop(max_outliers_df.index)

#-------------  Add new features -----------------------------------------------------

def feature_add(train_raw_data):
  ''' Add features

      Input: your objective data set
  
  '''
  train_raw_data['location'] = 1000* train_raw_data['longitude'] + train_raw_data['latitude']
  train_raw_data['location'] = train_raw_data['location'].round(0)


################################  CLASS  ################################

# Class for attribute transformer
# import important libray
from sklearn.base import BaseEstimator, TransformerMixin

class combined_attribute_adder_and_cleaner(BaseEstimator, TransformerMixin):
    '''data clean transfomer class'''
    
    def __init__(self, data_cleaner = False,set_threshold=False, threshold_value = 350 ): # no *args or **kargs
        # we need to set extra var to ensure do we need to purge the dataset. 
        # In my following experments, sometimes we don't need to do so. 
        self.data_clean = data_cleaner
        # Since we are using threshold from now on. This is crucial.
        self.set_threshold = set_threshold
        # threshold value
        self.threshold_value = threshold_value

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, data_df):
        # we first copy the data from our dataset.
        # operate on original data set sometimes is dangerous.
        X = data_df.copy()
        
        # Start to change all string columns that related to Dates to Time Series
        minutes = (X['acq_time'] % 100).astype(int)
        hours = (X['acq_time'] /100).astype(int) 
        
        #concatenate acq_data and acq_time
        X['acq_date'] =  X['acq_date'] + '-'  + hours.apply(str) + "-" + minutes.apply(str)
        X['acq_date'] = pd.to_datetime(X['acq_date'], format= '%Y-%m-%d-%H-%M')
        
        # This is a outdated version of transfering time series to index.
        # This didn't work so well when I training my model. 
        # So I comment this line. But sometimes, you want to do this.
        # X = X.set_index('acq_date')
        
        # Now we change our time series to a integer by indexing all instances with same month.
        X['month'] = X['acq_date'].dt.month
        
        # add new features. 
        feature_add(X)
        
        # one hot encoding
        # Transter categorical data to numerical value
        # Start to apply one hot encoding
        statelitte = pd.get_dummies(X.satellite)
        daynight = pd.get_dummies(X.daynight)
        X = pd.concat([X,statelitte.astype(int),daynight.astype(int)], axis=1)
        
        # drop all useless features and categorical features we alreayd transfered
        X = X.drop(['acq_time','acq_date', 'instrument','version','satellite','daynight','type','latitude','longitude'],axis=1) 

        
        # do outliers purgeing if it's needed
        
        if self.data_clean:
          # clean outliers
          X = find_outliers(X,'frp',drop=True,set_threshold=True,threshold_value=self.threshold_value)
            
          # clean negative frp instance
          negative_index = X[(X['frp'] <= 0)]  
          X = X.drop(negative_index.index)

          return X
        else:
        ## if we don't need to purge outliers, then just return X
          return X
        
    
#------------- transformer for output numpy array for our data set ---------------------

class numerical_transfer(BaseEstimator, TransformerMixin):
    '''numerical_transfer class'''
    
    def __init__(self, transfer_to_number=True ): # no *args or **kargs
        # This line seems pretty useless. But sometimes, we want this.
        self.transfer_to_number = transfer_to_number
        
    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X):
      # the values of a panda dataframe is all we need.
      if self.transfer_to_number:
        return X.values
      else:
        return X


#############################PIPE LINE###################################################


# Now we build a transformer to get all the above steps
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# purge pipeline is only for purgeing any outliers.
purge_pipeline = Pipeline([
        ('attribs_adder_cleaner', combined_attribute_adder_and_cleaner(data_cleaner=True)),
    ])

#number pipline will transfer all data to numpy array and do normailization
number_pipeline = Pipeline([
        ('num_transfer', numerical_transfer()),
        ('std_scaler', StandardScaler()),
    ])

# none purge pipeline will only get you the data with correct form, but don't delete any outliers
none_purge_pipeline = Pipeline([
        ('attribs_adder_cleaner', combined_attribute_adder_and_cleaner(data_cleaner=False))
    ])

# full pipeline to create a prepared dataset in a glance
full_pipeline = Pipeline([
     ('attribs_adder_cleaner', combined_attribute_adder_and_cleaner(data_cleaner=True)),
     ('num_transfer', numerical_transfer()),
     ('std_scaler', StandardScaler()),
])


############################FUNCTIONS ON PIPELINE##########################################

#-------------  create_trainning_data -----------------------------------------------------
# This function is for creating training data directly into numpy form
def create_trainning_data(data_df,num_array_needed = False):
    data_inner = data_df.copy()
    X = data_inner.drop("frp", axis=1) # drop labels for training set
    y = data_inner["frp"].copy()
    if num_array_needed: 
      X = number_pipeline.fit_transform(X)
      y = np.array(y.values)
    return X,y

#------------- create_dp_data -----------------------------------------------------

# This function is for creating a normalized training data but remain the panda dataset form
def create_dp_data(data): 
  target = data.pop('frp')
  normalized_df_test=(data-data.mean())/data.std()
  normalized_df_test['target'] = target
  normalized_df_test = normalized_df_test.drop(['latitude','longitude','type','Aqua','Terra','D','N'],axis=1)
  return normalized_df_test

