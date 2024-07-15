# Simulation code for the paper:
#
#   [1] Khac-Hoang Ngo, J. Ostman, G. Durisi, and A. Graell i Amat, “Secure aggregation is not private
#   against membership inference attacks,” in European Conference on Machine Learning and Principles
#   and Practice of Knowledge Discovery in Databases (ECML PKDD), Vilnius, Lithuania, Sep. 2024.
#   [Online]. Available: https://arxiv.org/pdf/2403.17775.
#
# written by Khac-Hoang Ngo (ngok@chalmers.se)

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import beta
from scipy.linalg import hadamard
from scipy.stats import norm
from multiprocessing import Pool
from scipy.optimize import root_scalar
from scipy.integrate import dblquad, quad
import matplotlib.pyplot as plt
import random
import pickle
import time
import math
import pdb
import csv
from functools import partial
import warnings
warnings.filterwarnings("ignore")

# Figure index. Incremented throughout the code
fig_idx = 0

# Function to load the data
def data_load(trial_idx,dim,num_samples,num_clients):
    # For a given trial index, this function returns the following:
    # a pair of input to be distinguished: (X1, X2)
    # samples of the equivalent noise: samples_Y 
    # the mean and covariance matrix of the equivalent noise: mean, cov

    # write the code that corresponds to your data :) 
    
    return X1, X2, samples_Y, mean, cov

# Compute the log PDF of a Gaussian random vector with given the mean and covariance matrix 
def logPDF_Gaussian(z,mean,cov):
    return np.log(multivariate_normal.pdf(z, mean=mean, cov=cov))

# Compute the log PDF of a Gaussian random vector with given the mean and eigendecomposition of the covariance matrix 
# We can choose to ignore eigenvalues that are too small
def logPDF_Gaussian_EVD(z,mean,D,V):
    return - 0.5*np.sum(((V.T@(z-mean))**2/D)*(D>1e-20)) 

# Compute relevant loglikelihood terms for the input pair (X1,X2), equivalent noise Y. 
def compute_loglikelihood(Y,X1,X2,f_loglikelihood):
    # Two possible mechanism outputs
    Z1 = Y + X1
    Z2 = Y + X2

    # loglikelihood for the correct input
    LL_Y1_X1 = f_loglikelihood(Y)
    LL_Y2_X2 = LL_Y1_X1

    # loglikelihood for the wrong input
    LL_Y1_X2 = f_loglikelihood(Z1 - X2)
    LL_Y2_X1 = f_loglikelihood(Z2 - X1)

    # save the results
    if np.all(([LL_Y1_X1, LL_Y1_X2, LL_Y2_X1, LL_Y2_X2])):
        idxvalid = 1
        return np.array([LL_Y1_X1, LL_Y1_X2, LL_Y2_X1, LL_Y2_X2, idxvalid])
    else:
        return np.array([0, 0, 0, 0, 0])

# Binomial proportion confidence interval: n trials, x successes
def bin_conf(x,n,conf_level,method):
    if x == n:
        return 1
    else:
        if method == 'ClopperPearson':
            return beta.ppf(1-(1-conf_level)/2, x+1, n-x)
        if method == 'Jeffreys':
            return beta.ppf(conf_level, 1/2+x, 1/2+n-x)
        if method == 'direct':
            return x/n

# Compute the FPR and FNR of the test
def f_FPR(thres,LL_Y1_X2,LL_Y1_X1,num_test,conf_level,method): 
    return bin_conf(np.sum(LL_Y1_X2 > LL_Y1_X1 - thres),num_test,conf_level,method) 

def f_FNR(thres,LL_Y2_X1,LL_Y2_X2,num_test,conf_level,method):
    return bin_conf(np.sum(LL_Y2_X1 > LL_Y2_X2 + thres),num_test,conf_level,method)
    # return np.mean(LL_Y2_X1 > LL_Y2_X2 + thres)

# Evaluate the privacy curve from the FPR and FNR of the test, according to Proposition 2 of [1]
def priv_audit(delta,FPR,FNR):
    # minimum and maximum test threshold

    # def FPR_lim(x):
    #     return np.abs(FPR(x) - 1 + delta)
    # def FNR_lim(x):
    #     return np.abs(FNR(x) - 1 + delta)
    # thres_max = minimize(FPR_lim, x0 = 1, method='Nelder-Mead').x
    # thres_min = minimize(FNR_lim, x0 = -1, method='Nelder-Mead').x

    thres_min = -1000
    thres_max = 1000

    print('thres_min = ', thres_min, '; thres_max = ', thres_max)

    if not np.isnan(thres_min) and not np.isnan(thres_max):
        # Audit epsilon according to Eq. (8) of [1]
        def f_eps(t):
            a = 1 - delta - FPR(t)
            b = 1 - delta - FNR(t)
            if a > 0 and b > 0:
                return max(np.log(a) - np.log(FNR(t)),
                    np.log(b) - np.log(FPR(t)))
            else:
                return -np.inf

        # Search for the best test threshold
        thres_range = np.linspace(thres_min, thres_max, num=1000)
        tmp0 = [f_eps(t) for t in thres_range]

        # print(thres_range[np.array(tmp0).argmax()])
        # print(thres_min,thres_max,thres_range[np.argmax(tmp0)], max(tmp0))

        # Plot if necessary
        # fpr = [FPR(t) for t in thres_range]
        # fnr = [FNR(t) for t in thres_range]

        # plt.figure
        # plt.plot(thres_range, tmp0)

        # plt.figure
        # plt.semilogy(thres_range, fpr)
        # plt.semilogy(thres_range, fnr,linestyle='dashed')
        # plt.grid()
        # plt.show()

        return max(tmp0)

if __name__ == "__main__":
    ###################### Parameters       
    dim = 7850  # model dimension
    num_samples = 5000  # number of samples
    num_clients = 60    # number of clients
    conf_level = .95    # confidence level in estimating the FPR and FNR
    bin_method = 'ClopperPearson' #'ClopperPearson', 'Jeffreys', or 'direct'

    num_trials = 5  # number of initial models

    delta_set = np.logspace(np.log10(.8*(1-beta.ppf(1-(1-conf_level)/2, 1, num_samples))), np.log10(1e-3), 20) # values of delta
    eps_set = np.zeros((np.size(delta_set),num_trials)) # initialize the values of audited epsilon

    # initialize the values of FNR and FPR for different test thresholds
    thres_range = np.linspace(-500,500,num=1000)
    fpr = np.zeros((np.size(thres_range),num_trials))
    fnr = np.zeros((np.size(thres_range),num_trials))

    ##################### Execution
    for trial_idx in range(num_trials):

        print("\n------- Trial ", trial_idx, " --------")

        ##################### Load the input pair (X1,X2) and samples of the equivalent noise Y 
        print("loading data...")
        start = time.time()
        (X1, X2, samples_Y, mean, cov) = data_load(trial_idx,dim,num_samples,num_clients)
        D, V = np.linalg.eigh(cov)

        # illustrate the input pair
        fig_idx += 1 
        plt.figure(fig_idx)
        plt.stem(X1,'--b')
        plt.stem(X2,':r')
        plt.title('Pair of inputs to distinguish')
        # plt.show()

        # illustrate the eigenvalues of the equivalent noise covariance matrix
        if trial_idx == 0:
            fig_idx += 1 
            plt.figure(fig_idx)
            plt.stem(D)
            plt.yscale("log")
            plt.title('Eigenvalues of the equivalent noise covariance matrix')

        stop = time.time() - start
        print("...in ", stop, "seconds")

        ####################### Evaluate FNR and FPR
        print("evaluating FPR and FNR...")
        start = time.time()
        
        # Function to compute the loglikelihood
        LL = partial(logPDF_Gaussian_EVD, mean = mean, D = D, V = V)
        
        ## OPTION 1: parallel computation
        num_processes = 10
        pool = Pool(processes=num_processes)
        partial_monte_carlo = partial(compute_loglikelihood, X1 = X1, X2 = X2, loglikelihood = LL)
        results = pool.map(partial_monte_carlo, samples_Y)
        pool.close()
        pool.join() 
        results = np.array(results).reshape(num_samples, -1)

        ## OPTION 2: serial computation
        # results = np.zeros((samples_Y.shape[0],5))
        # for iii in range(samples_Y.shape[0]):
        #     results[iii,:] = compute_loglikelihood(samples_Y[iii,:],X1, X2, LL)

        # Load the computation results
        LL_Y1_X1 = results[:, 0]
        LL_Y1_X2 = results[:, 1] 
        LL_Y2_X1 = results[:, 2] 
        LL_Y2_X2 = results[:, 3] 
        idxvalid = results[:, 4] 
        LL_Y1_X1 = LL_Y1_X1[idxvalid.astype(bool)]
        LL_Y1_X2 = LL_Y1_X2[idxvalid.astype(bool)]
        LL_Y2_X1 = LL_Y2_X1[idxvalid.astype(bool)]
        LL_Y2_X2 = LL_Y2_X2[idxvalid.astype(bool)]
        num_test = np.sum(idxvalid)

        # Compute the FPR and FNR
        FPR = partial(f_FPR,LL_Y1_X2=LL_Y1_X2,LL_Y1_X1=LL_Y1_X1,num_test=num_test,conf_level=conf_level,method=bin_method)
        FNR = partial(f_FNR,LL_Y2_X1=LL_Y2_X1,LL_Y2_X2=LL_Y2_X2,num_test=num_test,conf_level=conf_level,method=bin_method)

        stop = time.time() - start
        print("...in ", stop, "seconds")

        # Some illustrations of the FNR and FPR
        fpr_tmp = [FPR(t) for t in thres_range]
        fnr_tmp = [FNR(t) for t in thres_range]

        fig_idx += 1 
        plt.figure(fig_idx)
        plt.semilogy(thres_range, fpr_tmp,)
        plt.semilogy(thres_range, fnr_tmp,linestyle='dashed')
        plt.grid()
        plt.legend(['FPR', 'FNR'])

        fig_idx += 1 
        plt.figure(fig_idx)
        plt.loglog(fnr_tmp, fpr_tmp,)
        plt.grid()
        # plt.show()

        fnr[:,trial_idx] = np.array(fnr_tmp)
        fpr[:,trial_idx] = np.array(fpr_tmp)

        ################### Compute the privacy curve
        print("evaluating the privacy curve...")
        start = time.time()

        partial_monte_carlo = partial(priv_audit,FPR=FPR,FNR=FNR)

        ## OPTION 1: parallel computation
        pool = Pool(processes=num_processes)
        eps_set_tmp = pool.map(partial_monte_carlo, delta_set)
        pool.close()
        pool.join()

        ## OPTION 2: serial computation
        # eps_set_tmp = np.zeros_like(delta_set)
        # iii = 0
        # for delta in delta_set:
        #     eps_set_tmp[iii] = priv_audit(delta,FPR,FNR)
        #     iii += 1

        eps_set[:,trial_idx] = np.array(eps_set_tmp)

        stop = time.time() - start
        print("...in ", stop, "seconds")

    ####################### Compute the mean and standard deviations of the FNR, FPR and the privacy parameters
    eps_set_mean = np.mean(eps_set, axis=1).reshape((-1,1))
    std_devs = np.std(eps_set, axis=1).reshape((-1,1))

    fnr_mean = np.mean(fnr, axis=1).reshape((-1,1))
    fnr_std_devs = np.std(fnr, axis=1).reshape((-1,1))

    fpr_mean = np.mean(fpr, axis=1).reshape((-1,1))
    fpr_std_devs = np.std(fpr, axis=1).reshape((-1,1))

    delta_set = delta_set.reshape((-1,1))
    valid_indices = (np.array(eps_set_mean) >= 0).astype(bool)
    
    # Bound due to ClopperPearson
    bb = beta.ppf(1-(1-conf_level)/2, 1, num_samples)
    epsilon_CP_bound = np.log((1-delta_set-bb)/bb)

    ####################### Plot
    fig_idx += 1 
    plt.figure(fig_idx)
    plt.semilogy(eps_set_mean[valid_indices], delta_set[valid_indices], '*-r')
    plt.fill_betweenx(delta_set[valid_indices], eps_set_mean[valid_indices] - std_devs[valid_indices], eps_set_mean[valid_indices] + std_devs[valid_indices], alpha=0.3,color='red')
    plt.semilogy(epsilon_CP_bound, delta_set, 'black')
    plt.ylim(np.min(delta_set), 1)
    plt.xlabel(r'$\epsilon$', fontsize=14)
    plt.ylabel(r'$\delta$', fontsize=14)
    plt.grid()

    ####################### Save result
    # privacy parameters
    file_title = f'./results/priv_curve_n-{num_clients}.csv'
    f = open(file_title, 'w')
    writer = csv.writer(f)

    epsilon_CP_bound = epsilon_CP_bound.reshape((-1,))
    eps_set_mean = eps_set_mean.reshape((-1,))
    std_devs = std_devs.reshape((-1,))
    delta_set = delta_set.reshape((-1,))

    for ii in range(len(delta_set)):
        writer.writerow([epsilon_CP_bound[ii], eps_set_mean[ii] - std_devs[ii], eps_set_mean[ii], eps_set_mean[ii] + std_devs[ii], delta_set[ii]])
    f.close()

    # FPR and FNR
    file_title = f'./results/tradeoff_n-{num_clients}.csv'
    f = open(file_title, 'w')
    writer = csv.writer(f)

    fnr_mean = fnr_mean.reshape((-1,))
    fnr_std_devs = fnr_std_devs.reshape((-1,))
    fpr_mean = fpr_mean.reshape((-1,))
    fpr_std_devs = fpr_std_devs.reshape((-1,))

    for ii in range(len(fnr_mean)):
        writer.writerow([fnr_mean[ii] - fnr_std_devs[ii], fnr_mean[ii], fnr_mean[ii] + fnr_std_devs[ii], fpr_mean[ii] - fpr_std_devs[ii], fpr_mean[ii], fpr_mean[ii] + fpr_std_devs[ii]])
    f.close()

    plt.show()