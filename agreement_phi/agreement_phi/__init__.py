#!/usr/bin/env python3
import numpy as np
import itertools
import pymc3
from pymc3 import Model, Normal, Beta, HalfNormal, Uniform, find_MAP, Slice, Exponential, Constant, sample, math, Gamma, NUTS, HamiltonianMC, Metropolis
import pymc3 as pm
import scipy.stats as stats
from time import time
import pandas as pd

# Agreement Phi, see https://github.com/AlessandroChecco/agreement-phi
# Version 0.1.4

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
    
def minimal_matrix(v, fillval=np.nan):
    v = [doc[~np.isnan(doc)].tolist() for doc in v]
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def agreement(precision):
    return 1-2 * np.exp(-np.log(2)*precision/2)

def run_phi( data, **kwargs):
    data = np.array(data)
    
    # Check limits in **kwargs
    if kwargs.get( "limits" ) is not None: 
        limits = kwargs.get( "limits" )
    else:  
        limits = (np.nanmin(list(itertools.chain.from_iterable(data))),np.nanmax(list(itertools.chain.from_iterable(data))) )
    
    if kwargs.get("verbose") is not None:
        verbose = kwargs.get("verbose")
    else:
        verbose = False
        
    if kwargs.get("seed") is not None:
        seed = kwargs.get("seed")
    else:
        seed = 123
        
    if kwargs.get("table") is not None:
        table = kwargs.get("table")
    else:
        table = False
        
    if kwargs.get("N") is not None:
        N = kwargs.get("N")
    else:
        N = 1000

    if kwargs.get("keep_missing") is not None:
        keep_missing = kwargs.get("keep_missing")
    else:
        keep_missing = None #AUTO
        #keep_missing = True 
        
    if kwargs.get("fast") is not None:
        fast = kwargs.get("fast")
    else:
        fast = True
        
    if kwargs.get("njobs") is not None:
        njobs = kwargs.get("njobs")
    else:
        njobs = 2
        
    if kwargs.get("sd") is not None:
        sd = kwargs.get("sd")
    else:
        sd = 1000000
    
    # Check gt in **kwargs
    if kwargs.get( "gt" ) is not None: 
        gt = kwargs.get( "gt" )
    else:  
        gt = [None] * len(data)

    if verbose: print("Computing Phi")
    idx_of_gt = np.array([x is not None for x in gt])
    idx_of_not_gt = np.array([x is None for x in gt])
    num_of_gt = np.sum(idx_of_gt)
    
    
    basic_model = Model()
        
    for i,g in enumerate(gt):
        if g is not None:      
            gt[i] = scale_mat( np.array( [[gt[i]]* len(data[i])]), limits) [0][0]
    
    num_of_docs = len(data) # number of documents
    
    rectangular = True
    sparse = False
    if np.isnan(data).any():
        sparse = True
        data =  np.ma.masked_invalid(data)
        data = minimal_matrix(data)
    
    scaled = scale_mat(data, limits)
    
    if (np.count_nonzero(np.isnan(scaled))/scaled.size) > 0.2: # a lot of nans
        if verbose: print("WARNING: a lot of missing values: we are going to set keep_missing=False to improve convergence (if not manually overridden)")
        if keep_missing is None:
            keep_missing=False
    
    if (sparse and not keep_missing):
        rectangular = False
        scaled = [doc[~np.isnan(doc)].tolist() for doc in scaled] #make data a list of lists




    NUM_OF_ITERATIONS = N

        
    with basic_model:
        precision = Normal('precision',mu=2,sd=sd)
        #precision = Gamma('precision',mu=2,sd=1)
        
        if num_of_docs-num_of_gt == 1:
            mu = Normal('mu',mu=1/2,sd=sd)
        else:
            mu = Normal('mu',mu=1/2,sd=sd,shape=num_of_docs-num_of_gt)
        alpha = mu * precision
        beta = precision*(1-mu)
        
        if rectangular:
            masked = pd.DataFrame(scaled[idx_of_not_gt]) #needed to keep nan working
            if num_of_docs-num_of_gt == 1:
                Beta('beta_obs',observed=masked,alpha=alpha,beta=beta)
            else:
                Beta('beta_obs',observed=masked.T,alpha=alpha,beta=beta,shape=num_of_docs-num_of_gt)
        else:
            for i,doc in enumerate(scaled):
                Beta('beta_obs'+str(i),observed=doc,alpha=alpha[i],beta=beta[i])


        for i,g in enumerate(gt):
            if g is not None:
                mu = Normal('mu'+str(i),mu=gt[i],sd=1)
                alpha = mu * precision
                beta = precision*(1-mu)
                Beta('beta_obs_g'+str(i),observed=scaled[i],alpha=alpha,beta=beta)#alpha=a,beta=b,observed=beta)
    
    
    
    
        try:
            if fast:
                assert False
            stds = np.ones(basic_model.ndim)
            for _ in range(5):
                args = {'scaling': stds ** 2, 'is_cov': True}
                trace = pm.sample(round(NUM_OF_ITERATIONS/10), tune=round(NUM_OF_ITERATIONS/10), init=None, nuts_kwargs=args,chains=10,progressbar=verbose,random_seed=seed)
                samples = [basic_model.dict_to_array(p) for p in trace]
                stds = np.array(samples).std(axis=0)
    
            step = pm.NUTS(scaling=stds ** 2, is_cov=True, target_accept=0.9)
            start = trace[0]
            trace = sample(NUM_OF_ITERATIONS, tune=round(NUM_OF_ITERATIONS/2), njobs=njobs,chains=8, init=None, step=step, start=start,progressbar=verbose,random_seed=seed)    
            # Staistical inference
            beg = time()
            #start = find_MAP()
            bef_slice = time()
            #step = NUTS()# Metropolis()
            #step = Metropolis()
            aft_slice = time()
            bef_trace = time()
            #trace = sample(NUM_OF_ITERATIONS, progressbar=verbose,random_seed=123, njobs=njobs,start=start,step=step)
    #        trace = sample(NUM_OF_ITERATIONS, progressbar=verbose,random_seed=123, njobs=njobs,init=None,tune=100)     
        except:
            beg = time()
            step = Metropolis()
            #start = find_MAP()
            trace = sample(NUM_OF_ITERATIONS, progressbar=verbose,random_seed=seed, njobs=njobs,step=step)#,start=start)
        #pm.summary(trace,include_transformed=True)
        if np.float(pymc3.__version__) <= 3.3:
            res = pm.stats.df_summary(trace,include_transformed=True)
        else:
            res = pm.summary(trace,include_transformed=True)
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

        if table:
            col_names = res.columns[0:len(data)-1]
            for i, name in enumerate(col_names):
                l = len(scaled[i])
                for j in range(3):

                    b = res[name].iloc[j]
                    mu_res =  ( b * l - 0.5) / ( l - 1 )
                    res[name].iloc[j] =  np.clip( mu_res , 0, 1)*(limits[1]-limits[0])


        res["agreement"] = col_agreement
        res.insert(0, "precision", col_precision)
        aft_trace = time()
    computation_time = time()-beg
    if verbose: print("Elapsed time for computation: ",computation_time)
    
    convergence = True
    if np.isnan(res.loc['Rhat']['precision']) or np.abs(res.loc['Rhat']['precision'] - 1) > 1e-1:
        print("Warning! You need more iterations!")
        convergence = False
    if table:
        return {'agreement':col_agreement['mean'],'interval': col_agreement[['hpd_2.5','hpd_97.5']].values,"computation_time":computation_time,"convergence_test":convergence,'table':res}
    else:
        return {'agreement':col_agreement['mean'],'interval': col_agreement[['hpd_2.5','hpd_97.5']].values,"computation_time":computation_time,"convergence_test":convergence}

def main(args=None):
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Phi Agreement Measure")

    parser.add_argument("-f", "--file", dest="filename",
                        help="input FILE", metavar="FILE",required=True)
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose",
                        default=False,
                        help="don't print verbose messages")
    args = parser.parse_args()
    df = pd.read_csv(args.filename)
    if not df.applymap(lambda x: isinstance(x, (int, float))).all(1).all(0):
        raise ValueError('ERROR: csv is non numeric!')

    as_mat =  df.values
    result = run_phi( as_mat)
    print(result)
    
if __name__ == "__main__":
    main()
