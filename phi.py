import matplotlib.pyplot as plt
import seaborn as sns, numpy as np
import itertools
from alpha import krippendorff_alpha, interval_metric, nominal_metric
from pymc3 import Model, Normal, Beta, HalfNormal, Uniform, find_MAP, Slice, Exponential, Constant, sample, math, Gamma, NUTS, HamiltonianMC, Metropolis
import pymc3 as pm
import scipy.stats as stats
from time import time

# TODO
# Check degerate cases when using only one document

def scale_mat( mat, limits):   
    result = []
    for i,value in enumerate(mat):
        l = len(value)
        array = np.array(value)
        temp =  array - limits[0]
        temp = temp/(limits[1]-limits[0])
        temp = (temp * (l-1)+0.5)/l
        result.append(list(temp))
    return np.array(result,ndmin=2)

def agreement(precision):
    return 1-2 * np.exp(-np.log(2)*precision/2)

def run_phi( data, **kwargs):
    from IPython.display import display, HTML
    
    #assert False, "Stop"


    print("Computing Phi")
    
    data = np.array(data)
    
    # Check limits in **kwargs
    if kwargs.get( "limits" ) is not None: 
        limits = kwargs.get( "limits" )
    else:  
        limits = (np.min(list(itertools.chain.from_iterable(data))),np.max(list(itertools.chain.from_iterable(data))) )
    
    if kwargs.get("verbose") is not None:
        verbose = kwargs.get("verbose")
    else:
        verbose = False
        
    if kwargs.get("njobs") is not None:
        njobs = kwargs.get("njobs")
    else:
        njobs = 1
        
    if kwargs.get("sd") is not None:
        sd = kwargs.get("sd")
    else:
        sd = 1000000
    
    # Check gt in **kwargs
    if kwargs.get( "gt" ) is not None: 
        gt = kwargs.get( "gt" )
    else:  
        gt = [None] * len(data)
    
    idx_of_gt = np.array([x is not None for x in gt])
    idx_of_not_gt = np.array([x is None for x in gt])
    num_of_gt = np.sum(idx_of_gt)
    
    
    basic_model = Model()
        
    for i,g in enumerate(gt):
        if g is not None:      
            gt[i] = scale_mat( np.array( [[gt[i]]* len(data[i])]), limits) [0][0]
    
    num_of_docs = len(data) # number of documents
    
    scaled = scale_mat( data, limits)

    NUM_OF_INTERACTIONS = 1000
        
    with basic_model:
        precision = Normal('precision',mu=2,sd=sd)
        #precision = Gamma('precision',mu=2,sd=1)
        
        if num_of_docs-num_of_gt == 1:
            mu = Normal('mu',mu=1/2,sd=sd)
        else:
            mu = Normal('mu',mu=1/2,sd=sd,shape=num_of_docs-num_of_gt)
        alpha = mu * precision
        beta = precision*(1-mu)
        
        if num_of_docs-num_of_gt == 1:
            Beta('beta_obs',observed=scaled[idx_of_not_gt],alpha=alpha,beta=beta)
        else:
            Beta('beta_obs',observed=scaled[idx_of_not_gt].T,alpha=alpha,beta=beta,shape=num_of_docs-num_of_gt)#alpha=a,beta=b,observed=beta)
#        Beta('beta_obs',observed=scaled,alpha=alpha,beta=beta,shape=num_of_docs)#alpha=a,beta=b,observed=beta)

        for i,g in enumerate(gt):
            if g is not None:
                mu = Normal('mu'+str(i),mu=gt[i],sd=1)
                alpha = mu * precision
                beta = precision*(1-mu)
                Beta('beta_obs'+str(i),observed=scaled[i],alpha=alpha,beta=beta)#alpha=a,beta=b,observed=beta)
    
        # Staistical inference
        start = find_MAP()
        bef_slice = time()
        step = Metropolis()
        aft_slice = time()
        bef_trace = time()
        #trace = sample(NUM_OF_INTERACTIONS, step=step, progressbar=True, start=start,radom_seed=123)
        trace = sample(NUM_OF_INTERACTIONS, progressbar=verbose,random_seed=123, njobs=njobs,start=start,step=step)
        pm.summary(trace,include_transformed=True)
        res = pm.stats.df_summary(trace,include_transformed=True)
        
        res.drop(["sd","mc_error"], axis=1, inplace = True)
        res = res.transpose()
        res["agreement"] = agreement(res['precision']) 


        # ---- 
        
        #sub_res = res.copy()


        # Mu rescaling

        col_agreement = res["agreement"]
        col_precision = res["precision"]

        res.drop("agreement", inplace = True, axis = 1)
        res.drop("precision", inplace = True, axis = 1)

        col_names = res.columns

        for i, name in enumerate(col_names):
            l = len(scaled[i])
            for j in range(3):

                b = res[name].iloc[j]
                mu_res =  ( b * l - 0.5) / ( l - 1 )
                res[name].iloc[j] =  np.clip( mu_res , 0, 1)


        res["agreement"] = col_agreement
        res.insert(0, "precision", col_precision)


        
        aft_trace = time()
    return res

