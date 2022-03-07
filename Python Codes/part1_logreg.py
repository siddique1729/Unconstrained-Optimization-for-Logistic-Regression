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
a.head

# copying the dataset for cleaning

to_clean = a.copy()

# number of 0's in respective columns having length dimensions
# ApproachWidth_M, StructureLength_M, RoadwayWidth_M, DECK_WIDTH_M

count1 = (to_clean['ApproachWidth_M'] == 0).sum()
count2 = (to_clean['StructureLength_M'] == 0).sum()
count3 = (to_clean['RoadwayWidth_M'] == 0).sum()
count4 = (to_clean['DECK_WIDTH_M'] == 0).sum()

print(count1,count2,count3,count4)

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
sats = (to_clean['br_cond']==0).sum() # bridges with no reconstruction
unsats = 1000 - sats

print("Bridges with Satisfactory Condition:",sats)
print("Bridges with Unsatisfactory Condition:",unsats)


# Checking the types of columns
to_clean.info()
to_clean.head()


# Selecting feature variables

xy = to_clean[['recon','age','age_sq','PPTIN','DeckArea_SQM','ADT','br_cond']]
xy.head()

xy.hist(bins = 50, figsize=(20,15))
plt.show()


# Normalizing feature variables

xy['age'] = (xy['age']-np.min(xy['age']))/(np.max(xy['age'])-np.min(xy['age']))
xy['age_sq'] = (xy['age_sq']-np.min(xy['age_sq']))/(np.max(xy['age_sq'])-np.min(xy['age_sq']))
xy['PPTIN'] = (xy['PPTIN']-np.min(xy['PPTIN']))/(np.max(xy['PPTIN'])-np.min(xy['PPTIN']))
xy['DeckArea_SQM'] = (xy['DeckArea_SQM']-np.min(xy['DeckArea_SQM']))/(np.max(xy['DeckArea_SQM'])-np.min(xy['DeckArea_SQM']))
xy['ADT'] = (xy['ADT']-np.min(xy['ADT']))/(np.max(xy['ADT'])-np.min(xy['ADT']))

# Converting the columns into float class by converting them into numerics

xy["recon"] = pd.to_numeric(xy["recon"])
xy["age"] = pd.to_numeric(xy["age"])
xy["age_sq"] = pd.to_numeric(xy["age_sq"])
xy["PPTIN"] = pd.to_numeric(xy["PPTIN"])
xy["ADT"] = pd.to_numeric(xy["ADT"])
xy["DeckArea_SQM"] = pd.to_numeric(xy["DeckArea_SQM"])
xy["br_cond"] = pd.to_numeric(xy["br_cond"])

# Optimization using Nelder-Mead method

y = xy.iloc[:,6] # output variable
x = xy.iloc[:,[0,1,2,3,4,5]] # input variable

# Logistic Function to optimize

def bridge(beta,x,y):
    z = beta[0] + beta[1]*x.iloc[:,0] + beta[2]*x.iloc[:,1] + beta[3]*x.iloc[:,2] + beta[4]*x.iloc[:,3]+ beta[5]*x.iloc[:,4] + beta[6]*x.iloc[:,5]
    prob = 1/(1+np.exp(-z)) # probability of the function
    LL = np.sum(y*np.log(prob) + (1-y)*np.log(1-prob)) # log-likelihood
    LL = (-1)*LL # maximizing log-likelihood
    return(LL)

# Performing optimization, providing init coefficient values

beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
loglik = minimize(bridge, beta, args=(x,y,), method='Nelder-Mead',options={"maxiter":5000})
loglik
loglik.x

# Optimization using Newton's method, refer to the bridge function

beta = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
tol = 1e-05 # tolerance limit
err_diff = 100000 # difference
itr = 0 # nth number of iterations

while(err_diff>tol):
    a = nd.Gradient(bridge)(beta,x,y)
    b = nd.Hessian(bridge)(beta,x,y)
    binv = np.linalg.inv(b)
    eta = np.matmul(a,binv)
    beta_est = beta - eta
    err_diff = np.sum(np.sqrt((beta_est-beta)**2))
    err_diff = abs(err_diff)
    beta = beta_est
    itr = itr + 1
    

print(beta)

# computing eigen values of the hessian
# e = spl.eig
# Eigene value of the Hessian Matrix

ww,vv = np.linalg.eig(b)
print(ww)

if ww[0]>0 and ww[1]>0 and ww[2]>0 and ww[3]>0 and ww[4]>0 and ww[5]>0 and ww[6]>0:
    print("It is Convex")
elif ww[0]<0 and ww[1]<0 and ww[2]<0 and ww[3]<0 and ww[4]<0 and ww[5]<0 and ww[6]<0:
    print("It is Concave")
else:
    print("It is neither concave nor convex")
    
    
# Convex, global minima; Concave, global maxima