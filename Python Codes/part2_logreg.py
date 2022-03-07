"""
# The code is written by Jawwad Shadman Siddique | R11684947
# Date of Submission: 03 / 07 / 2022
"""

# importing required libraries & packages

import os
import pandas as pd 
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from random import choices
from scipy.optimize import minimize
import scipy.stats as st

# Checking Directory

os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()

# Reading the dataset

a = pd.read_csv('dat.csv')

# checking whether there are any missing values
a.isnull().any()
# calculating the missing values per column
a.isnull().sum()


# checking the total number of data (rows & columns)
a.shape

# copying the dataset for cleaning

to_clean = a.copy()

# number of 0's in respective columns having length dimensions
# ApproachWidth_M, StructureLength_M, RoadwayWidth_M, DECK_WIDTH_M

count1 = (to_clean['ApproachWidth_M'] == 0).sum()
count2 = (to_clean['StructureLength_M'] == 0).sum()
count3 = (to_clean['RoadwayWidth_M'] == 0).sum()
count4 = (to_clean['DECK_WIDTH_M'] == 0).sum()

# Creating a reconstruction column

to_clean['recon'] = np.where(to_clean['YEAR_RECON'] == 0, 0, 1)
(to_clean['recon']==0).sum() # bridges with no reconstruction

# Creating age column

to_clean['age'] = np.where(to_clean['YEAR_RECON'] > to_clean['YearBuilt'],
                           to_clean['YEAR_RECON'], to_clean['YearBuilt'])

to_clean['age'] = 2022 - to_clean['age']

# Creating age square column

to_clean['age_sq'] = to_clean['age']**2

# Bridge condition column

to_clean['br_cond'] = np.where(to_clean['BridgeCond']=='S',1,0)

# checking the cleaned dataset
to_clean.head()


# Selecting feature variables

xy = to_clean[['recon','age','age_sq','PPTIN','DeckArea_SQM','ADT','br_cond']]
xy.head()


# Normalizing feature variables

xy['age'] = (xy['age']-np.min(xy['age']))/(np.max(xy['age'])-np.min(xy['age']))
xy['age_sq'] = (xy['age_sq']-np.min(xy['age_sq']))/(np.max(xy['age_sq'])-np.min(xy['age_sq']))
xy['PPTIN'] = (xy['PPTIN']-np.min(xy['PPTIN']))/(np.max(xy['PPTIN'])-np.min(xy['PPTIN']))
xy['DeckArea_SQM'] = (xy['DeckArea_SQM']-np.min(xy['DeckArea_SQM']))/(np.max(xy['DeckArea_SQM'])-np.min(xy['DeckArea_SQM']))
xy['ADT'] = (xy['ADT']-np.min(xy['ADT']))/(np.max(xy['ADT'])-np.min(xy['ADT']))

xy.head()


# Converting the columns into float class by converting them into numerics

xy["recon"] = pd.to_numeric(xy["recon"])
xy["age"] = pd.to_numeric(xy["age"])
xy["age_sq"] = pd.to_numeric(xy["age_sq"])
xy["PPTIN"] = pd.to_numeric(xy["PPTIN"])
xy["ADT"] = pd.to_numeric(xy["ADT"])
xy["DeckArea_SQM"] = pd.to_numeric(xy["DeckArea_SQM"])
xy["br_cond"] = pd.to_numeric(xy["br_cond"])

# Start: Random sampling with repetition

# Declaring empty lists for coefficients for each sampling

b0 = []
b1 = []
b2 = []
b3 = []
b4 = []
b5 = []
b6 = []


for i in range(0,5000,1):
    xyy = xy.sample(replace=True)
    
    # Optimization using Nelder-Mead method
    
    y = xyy.iloc[:,6] # output variable
    x = xyy.iloc[:,[0,1,2,3,4,5]] # input variable
    
    # Logistic Function to optimize
    
    def bridge(beta,x,y):
        z = beta[0] + beta[1]*x.iloc[:,0] + beta[2]*x.iloc[:,1] + beta[3]*x.iloc[:,2] + beta[4]*x.iloc[:,3]+ beta[5]*x.iloc[:,4] + beta[6]*x.iloc[:,5]
        prob = 1/(1+np.exp(-z)) # probability of the function
        LL = np.sum(y*np.log(prob) + (1-y)*np.log(1-prob))
        LL = (-1)*LL # maximizing log-likelihood
        return(LL)
    
    # Performing optimization, providing init coefficient values
    
    beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    loglik = minimize(bridge, beta, args=(x,y,), method='Nelder-Mead',options={"maxiter":5000})
    
    # Storing the coefficient values
    
    b0.append(loglik.x[0])
    b1.append(loglik.x[1])
    b2.append(loglik.x[2])
    b3.append(loglik.x[3])
    b4.append(loglik.x[4])
    b5.append(loglik.x[5])
    b6.append(loglik.x[6])
    

# Creating dataframe for the coefficients of all samples

data_dic = {'b0':b0, 'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5,'b6':b6}
beta_dat = pd.DataFrame(data_dic)

#print(beta_dat)
beta_dat.head
beta_dat.shape


b0dat = data_dic['b0']
b1dat = data_dic['b1']
b2dat = data_dic['b2']
b3dat = data_dic['b3']
b4dat = data_dic['b4']
b5dat = data_dic['b5']
b6dat = data_dic['b6']

# Computing 95% confidence interval for the coefficients for 5000 samples



interval_b0 = st.t.interval(alpha=0.95, df=len(b0dat)-1, loc=np.mean(b0dat), scale=st.sem(b0dat))
interval_b1 = st.t.interval(alpha=0.95, df=len(b1dat)-1, loc=np.mean(b1dat), scale=st.sem(b1dat))
interval_b2 = st.t.interval(alpha=0.95, df=len(b2dat)-1, loc=np.mean(b2dat), scale=st.sem(b2dat))
interval_b3 = st.t.interval(alpha=0.95, df=len(b3dat)-1, loc=np.mean(b3dat), scale=st.sem(b3dat))
interval_b4 = st.t.interval(alpha=0.95, df=len(b4dat)-1, loc=np.mean(b4dat), scale=st.sem(b4dat))
interval_b5 = st.t.interval(alpha=0.95, df=len(b5dat)-1, loc=np.mean(b5dat), scale=st.sem(b5dat))
interval_b6 = st.t.interval(alpha=0.95, df=len(b6dat)-1, loc=np.mean(b6dat), scale=st.sem(b6dat))


print(interval_b0)
print(interval_b1)
print(interval_b2)
print(interval_b3)
print(interval_b4)
print(interval_b5)
print(interval_b6)
