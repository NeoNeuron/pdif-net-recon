import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree
# import pandas as pd
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import norm
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans


def embedding(x, y, mx, my, lag = 1, h = 1):
    ## Embedding in state space: the first mx columns of pts are the
    ## x-embedding the next my columns are the y-embedding and the final columns
    ## are the horizon values
    m = max(mx, my)
    n = len(x)
    leng = lag * (m - 1)
    pts = np.zeros((n - leng - h, mx + my + 2))
    pts[:, mx + my] = x[h + leng:]
    pts[:, mx + my + 1] = y[h + leng:]
    for ii in range(1, mx + 1):
        pts[:, mx - ii] = x[(lag*(m - ii)):(n - lag*(ii-1) - h)]
    for ii in range(1, my + 1):
        pts[:, mx + my - ii] = y[(lag*(m - ii)):(n - lag*(ii-1) - h)]
    ##
    ## As there are potentially NaNs in the data, we need to remove these
    ## embeddings. However, if we had done this earlier, we might have removed
    ## valid points in the embedding space (e.g. x = [0, 1, 2, 3, nan, 4, 5, 6]
    ## should have embedding vectors [0, 1, 2], [1, 2, 3], [4, 5, 6] but an
    ## earlier removal of [2, 3, nan] from x might only leave the embedding
    ## vector [4, 5, 6] - this is a problem in other code that does not
    ## consider NaNs)
    pts = pts[~np.any(np.isnan(pts), axis = 1), :]
    if h <= 0:
        pts = pts[:, :(mx + my)]
    return pts


def convergent_cross_mapping_full(x, y, mx = 2, my = 2, rho_tol = 0.05, \
        exp_fit = False, min_T=1000, n_T = 20, \
        plot = False, random = True, n_samples = 20, \
        metric = 'minkowski', seed = 2212):
    ##
    ## if output is anything other than 'exp_fit' (as in Monster et al [2016])
    ## or then it is treated as in Clark et al [2015] e.g. convergence if
    ## increase in rho between minimum and maximum library size > rho_tol
    ## n_T is the number of points on curve
    ## if n_samples = None, then the maximum number of samples are used
    ## plot = True shows a plot of the correlation rho against T values
    ##
    np.random.seed(seed)
    ##
    ## The initial set up is the same as in almost all other indices, but like
    ## similarity indices doesn't require x_{t + 1}, y_{t + 1}
    m = max(mx, my)
    pts = embedding(x, y, mx = mx, my = my, h = 0)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    ##
    if pts.shape[0] == 0:
        return np.nan, np.nan
    ##
    ## One aim here is to make this function as minimally computationally
    ## intensive as possible whilst still accurate
    ## One way we can do that is to be smart about which values of T to include
    ## in the computation so that we can show convergence of the CCM without
    ## unnecessary computation
    ## How can we achieve that sensibly?
    ## (a) If we want to fit a curve (i.e. output = 'exp_fit'), then compute
    ##  rho for T = T_0,...,T_m, where T_m = n, T_0 = m + 2
    ##  We could space these T values out equidistantly but better
    ##  (computationally and for fitting) if more weight given to smaller
    ##  values of T, since the function may plateau (and so exponential curve
    ##  fit doesn't converge)
    ##  Not clear how to do this. For now we take T_x = k * a ** x
    ##  We need to find a suitable k and a, e.g.
    ##  a = (T_m - T_0) ** (1 / (n_T - 1)), k = (T_m - T_0) / (a ** n_T)
    ##  seems to work reasonably well (sort of like 2 ** n but optimally scaled)
    ## (b) If we just want to see if there is an increase between T_m = n and
    ##  T_0 = m + 2, and we don't want to plot, then only do these two values
    ##  Unless we have (rho_n - rho_0 > rho_tol), we assume convergence is not
    ##  satisfied and ccm = 0
    ##  Note: this is the strategy in Clark et al [2015], who stop here -
    ##  they use rho_tol = 0
    ##
    ##
    def cm_rho_fun(pts_x, pts_y, m = m, metric = metric):
        '''
        Predicting pts_y using embedding information of pts_x
        '''
        ##
        xTree = KDTree(pts_x, metric = metric)
        ##
        ## dimension E = m, need m + 1 nearest neighbours for a bounding simplex
        ## (not including itself, hence k = m + 2)
        dists_x, idxs_x = xTree.query(pts_x, k = m + 2)
        ##
        u_x = np.exp(- dists_x[:, 1:] / (dists_x[:, 1].reshape(-1, 1) + 1e-5 ))
        null_ind = dists_x[:,1] == 0
        u_x[null_ind,:] = dists_x[null_ind, 1:] == 0
        ##
        w_x = u_x / u_x.sum(axis = 1).reshape(-1, 1)
        ##
        # y_hat = (y[idxs_x[:, 1:]] * w_x).sum(axis = 1)
        # cm_rho = np.corrcoef(y_hat, y)[0, 1]
        y_hat = np.empty((pts_y.shape[0], pts_y.shape[1]))
        cm = np.empty(pts_y.shape[1])
        for i in range(pts_y.shape[1]):
            y_hat[:,i] = (pts_y[:,i][idxs_x[:, 1:]] * w_x).sum(axis = 1)
            cm[i] = np.corrcoef(y_hat[:,i], pts_y[:,i])[0, 1]
        ##
        cm_rho = np.mean(cm)
        return cm_rho
        ##
    ##
    def cm_rhos(pts, m = m, metric = metric):
        params = {'m': m, 'metric': metric}
        # cm_xy = cm_rho_fun(pts[:, :mx], pts[:, mx + my - 1], **params)
        # cm_yx = cm_rho_fun(pts[:, mx:], pts[:, mx - 1], **params)
        cm_xy = cm_rho_fun(pts[:, :mx], pts[:, mx:], **params)
        cm_yx = cm_rho_fun(pts[:, mx:], pts[:, :mx], **params)
        return cm_yx, cm_xy
        ##
    ##
    min_T = max(m + 2, min_T)
    max_T = n
    ##
    ## Library lengths: list_T
    if plot is True or exp_fit is True:
        a = (max_T - min_T) ** (1 / (n_T - 1)) + 1e-8
        k = (max_T - min_T) / (a ** n_T)
        fun_T = lambda x: k * a ** x
        ##
        list_T = fun_T(np.arange(n_T) + 1) / fun_T(n_T) * (max_T - min_T)
        list_T = min_T + list_T.astype(int)
        # list_T = np.linspace(min_T, max_T, n_T).astype(int)
    else:
        list_T = np.array([min_T, max_T])
    ##
    ## Number of samples for each library length
    n_samples_in = n_samples
    n_samples = n // list_T
    if n_samples_in is not None:
        n_samples = \
            np.min((n_samples, np.repeat(n_samples_in, len(list_T))), axis = 0)
    ##
    # print(n_samples,list_T)
    ## Computation
    cm_vals = np.zeros((len(list_T), 2))
    for ii in range(len(list_T)):
        for jj in range(n_samples[ii]):
            if random:
                if n == list_T[ii]:
                    ind = np.arange(0,n)
                else:
                    ind = np.random.choice(n, size=list_T[ii], replace=False)
            else:
                ind = np.arange(jj * list_T[ii],(jj+1)* list_T[ii])
            # print(pts[ind, :].shape)
            cm_vals[ii, :] += cm_rhos(pts[ind, :])
        cm_vals[ii, :] /= n_samples[ii]
    ##
    ## If exponential curve fit (as in Monster et al [2016]):
    if exp_fit is True:
        def exp_fun(x, gamma, p_0, p_inf):
            return (p_0 - p_inf) * np.exp(- gamma * x) + p_inf
        p0 = (1e-6, 0, 0.5)
        try:
            par_yx, _ = curve_fit(exp_fun, list_T, cm_vals[:, 0], p0 = p0)
            if np.median(cm_vals[:-1, 0]) - \
                    exp_fun(list_T[0], *par_yx) < rho_tol or exp_fun(list_T[-1], *par_yx) \
                         < rho_tol or par_yx[-1] > 1:
                ccm_yx = cm_vals[-1, 0]
            else:
                ccm_yx = par_yx[-1]
        except:
            ccm_yx = np.nan
        try:
            par_xy, _ = curve_fit(exp_fun, list_T, cm_vals[:, 1], p0 = p0)
            ##
            if np.median(cm_vals[:-1, 1]) - \
                    exp_fun(list_T[0], *par_xy) < rho_tol or exp_fun(list_T[-1], *par_xy) \
                         < rho_tol or par_xy[-1] > 1:
                ccm_xy = cm_vals[-1, 1]
            else:
                ccm_xy = par_xy[-1]
        except:
            ccm_xy = np.nan
    else:
        if cm_vals[-1, 0] - max(cm_vals[:-1, 0]) < 0 or cm_vals[-1, 0] < 0:
            ccm_yx = 0
        else:
            ccm_yx = cm_vals[-1, 0]
        if cm_vals[-1, 1] - max(cm_vals[:-1, 1]) < 0 or cm_vals[-1, 1] < 0:
            ccm_xy = 0
        else:
            ccm_xy = cm_vals[-1, 1]
    ##
    if plot is True:
        plt.figure()
        plt.plot(list_T, cm_vals[:, 1], 'bo')
        plt.plot(list_T, cm_vals[:, 0], 'ro')
        if exp_fit:
            xx = np.arange(1, max_T)
            try:
                plt.plot(xx, exp_fun(xx, *par_xy), 'b')
                plt.plot(xx, exp_fun(xx, *par_yx), 'r')
            except:
                pass
    ##
    ## ccm_xy is the convegent cross-mapping of x given y e.g. CCM(X|Y) / CCM_{Y->X}
    ## Note if y causally influences x, y influences dynamics of x and knowledge
    ## of shadow manifold Ax can be used to estimate y. Hence
    ## CCM(X|Y) = rho(y^, y)
    return ccm_xy, ccm_yx
    ## end function for CCM



from scipy.signal import stft

def freq_convergent_cross_mapping(x, y, ws, hop, fs=1, rho_tol=0.05, \
        exp_fit = False, min_T=1000, n_T = 20, \
        plot = False, random = True, n_samples = 20, \
        metric = 'minkowski', truncate_tol=1e-2, seed = 2212):

    _, _, spec_x = stft(x, fs, nperseg=ws, noverlap=ws-hop,nfft=ws, boundary=None)
    _, _, spec_y = stft(y, fs, nperseg=ws, noverlap=ws-hop,nfft=ws, boundary=None)
    Mx = abs(spec_x).transpose()
    My = abs(spec_y).transpose()

    pts = embedding(x[::hop], y[::hop], mx = ws // hop, my = ws // hop, h = 0)
    n = pts.shape[0]
    x_cols = list(range(ws // hop))
    y_cols = list(range(ws // hop, 2*(ws // hop)))    

    def truncate(data, tol=truncate_tol):
        # data (L, D)
        std = data.std(axis=0)
        return data[:,std>tol]

    X, Y = x[ws-1:][::hop], y[ws-1:][::hop]
    n = X.shape[0]
    Mx, My = truncate(Mx)[:n], truncate(My)[:n]
    m = max(Mx.shape[1], My.shape[1])
    # print('M:',m)


    def cm_rho_fun(pts_x, delay_y, m = m, metric = metric):
        ##
        xTree = KDTree(pts_x, metric = metric)
        ##
        ## dimension E = m, need m + 1 nearest neighbours for a bounding simplex
        ## (not including itself, hence k = m + 2)
        dists_x, idxs_x = xTree.query(pts_x, k = m + 2)
        ##
        u_x = np.exp(- dists_x[:, 1:] / (dists_x[:, 1].reshape(-1, 1) + 1e-5 ))
        # u_x = np.exp(- dists_x[:, 1:] / dists_x[:, 1].reshape(-1, 1))
        ##
        w_x = u_x / u_x.sum(axis = 1).reshape(-1, 1)
        ##
        y_hat = np.empty((delay_y.shape[0], delay_y.shape[1]))
        cm = np.empty(delay_y.shape[1])
        # print(pts_x.shape, pts_y.shape, w_x.shape)
        for i in range(delay_y.shape[1]):
            # print(i,':',pts_y[:,i].shape,idxs_x[:, 1:].shape)
            y_hat[:,i] = (delay_y[:,i][idxs_x[:, 1:]] * w_x).sum(axis = 1)
            cm[i] = np.corrcoef(y_hat[:, i], delay_y[:, i])[0, 1]
        ##
        # print(pts_x.shape,y.shape)
        cm_rho = np.mean(cm)
        return cm_rho
    def cm_rhos(Mx, My, delay_x, delay_y, m = m, metric = metric):
        params = {'m': m, 'metric': metric}
        cm_xy = cm_rho_fun(Mx, delay_y, **params)
        cm_yx = cm_rho_fun(My, delay_x, **params)
        return cm_yx, cm_xy


    min_T = max(m + 2, min_T)
    max_T = n
    ##
    ## Library lengths: list_T
    if plot is True or exp_fit is True:
        a = (max_T - min_T) ** (1 / (n_T - 1)) + 1e-8
        k = (max_T - min_T) / (a ** n_T)
        fun_T = lambda x: k * a ** x
        ##
        list_T = fun_T(np.arange(n_T) + 1) / fun_T(n_T) * (max_T - min_T)
        list_T = min_T + list_T.astype(int)
    else:
        list_T = np.array([min_T, max_T])
    ##
    ## Number of samples for each library length
    n_samples_in = n_samples
    n_samples = n // list_T
    if n_samples_in is not None:
        n_samples = \
            np.min((n_samples, np.repeat(n_samples_in, len(list_T))), axis = 0)

    ## Computation
    cm_vals = np.zeros((len(list_T), 2))
    for ii in range(len(list_T)):
        for jj in range(n_samples[ii]):
            if random:
                if n == list_T[ii]:
                    ind = 0
                else:
                    ind = np.random.choice(n - list_T[ii])
            else:
                ind = jj * list_T[ii]
            cm_vals[ii, :] += cm_rhos(Mx[ind:ind + list_T[ii], :], My[ind:ind + list_T[ii], :], \
                pts[ind:ind + list_T[ii], x_cols], pts[ind:ind + list_T[ii], y_cols], m, metric)
        cm_vals[ii, :] /= n_samples[ii]
    ##
    ## If exponential curve fit (as in Monster et al [2016]):
    if exp_fit is True:
        def exp_fun(x, gamma, p_0, p_inf):
            return (p_0 - p_inf) * np.exp(- gamma * x) + p_inf
        p0 = (1e-6, 0, 0.5)
        try:
            par_yx, _ = curve_fit(exp_fun, list_T, cm_vals[:, 0], p0 = p0)
            if np.median(cm_vals[:-1, 0]) - \
                    exp_fun(list_T[0], *par_yx) < rho_tol or exp_fun(list_T[-1], *par_yx) \
                        < rho_tol or par_yx[-1]>1:
                ccm_yx = cm_vals[-1, 0]
            else:
                ccm_yx = par_yx[-1]
        except:
            ccm_yx = np.nan
        try:
            par_xy, _ = curve_fit(exp_fun, list_T, cm_vals[:, 1], p0 = p0)
            ##
            if np.median(cm_vals[:-1, 1]) - \
                    exp_fun(list_T[0], *par_xy) < rho_tol or exp_fun(list_T[-1], *par_xy) \
                        < rho_tol or par_xy[-1]>1:
                ccm_xy = cm_vals[-1, 1]
            else:
                ccm_xy = par_xy[-1]
        except:
            ccm_xy = np.nan
    else:
        if cm_vals[-1, 0] - max(cm_vals[:-1, 0]) < rho_tol or cm_vals[-1, 0] < 0:
            ccm_yx = 0
        else:
            ccm_yx = cm_vals[-1, 0]
        if cm_vals[-1, 1] - max(cm_vals[:-1, 1]) < rho_tol or cm_vals[-1, 1] < 0:
            ccm_xy = 0
        else:
            ccm_xy = cm_vals[-1, 1]
    ##
    if plot is True:
        plt.figure()
        plt.plot(list_T, cm_vals[:, 1], 'bo')
        plt.plot(list_T, cm_vals[:, 0], 'ro')
        if exp_fit:
            xx = np.arange(1, max_T)
            try:
                plt.plot(xx, exp_fun(xx, *par_xy), 'b')
                plt.plot(xx, exp_fun(xx, *par_yx), 'r')
            except:
                pass
    ##
    ## ccm_xy is the frequency CCM of x given y e.g. CCM(X|Y) / CCM_{Y->X}
    ## Note if y causally influences x, y influences dynamics of x and knowledge
    ## of shadow manifold Ax can be used to estimate y. Hence
    ## FCCM(X|Y) = rho(y^, y)
    return ccm_xy, ccm_yx
    ## end function for CCM

import math
import itertools
# from collections import Counter
# from scipy.stats import entropy
from sklearn.metrics import adjusted_mutual_info_score as AMI
def symbolic_convergent_cross_mapping(x, y, mx = 2, my = 2, rho_tol = 0.05, \
        exp_fit = False, min_T=1000, n_T = 20, \
        plot = False, random = True, n_samples = 20, \
        metric = 'minkowski', seed = 2212):

    np.random.seed(seed)
    ##
    ## The initial set up is the same as in almost all other indices, but like
    ## similarity indices doesn't require x_{t + 1}, y_{t + 1}
    m = max(mx, my)

    sym_dict, s_ind = {}, 1
    for it in itertools.permutations(range(m), m):
        sym_dict[''.join([str(elem) for elem in it])] = s_ind
        s_ind += 1

    pts = embedding(x, y, mx = mx, my = my, h = 0)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))

    def convert_symbolic_rank(x):
        ranks = np.empty_like(x)
        sort_indice = np.argsort(x, kind='mergesort')    
        ranks[sort_indice] = np.arange(0,len(x))
        return sym_dict[''.join(map(str, [int(ranks[j]) for j in range(len(ranks))]))]

    # def factor(x):
    #     return math.factorial(x)

    # def expected_MI(U, V, N: int):
    #     EMI = 0
    #     # print(U)
    #     # print(V)
    #     # return np.log(N+len(U)*len(V)-len(U)-len(V))-np.log(N-1)
    #     for i in range(len(U)):
    #         for j in range(len(V)):
    #             a, b = U[i], V[j]
    #             EMI += a*b/(N**2)*np.log(N*(a-1)*(b-1)/((N-1)*a*b) + N/a/b)
    #             # for n_ij in range(max(a+b-N,1),min(a,b)+1):
    #             #     EMI += (n_ij/N) * np.log((N*n_ij/a/b)*(factor(a)*factor(b)*factor(N-a)*factor(N-b)) \
    #             #         /(factor(N)*factor(n_ij)*factor(a-n_ij)*factor(b-n_ij)*factor(N-a-b+n_ij)))
    #     return EMI

    # def joint_rankvectors(s1, s2):
    #     N = len(s1)
    #     assert N==len(s2)
    #     return [s1[i]+s2[i] for i in range(N)]

    def cm_rho_fun(pts_x, pts_y, m = m, metric = metric):

        s_y = [convert_symbolic_rank(pts_y[i]) for i in range(pts_y.shape[0])]
        ##
        xTree = KDTree(pts_x, metric = metric)
        ##
        ## dimension E = m, need m + 1 nearest neighbours for a bounding simplex
        ## (not including itself, hence k = m + 2)
        dists_x, idxs_x = xTree.query(pts_x, k = m + 2)
        ##
        # u_x = np.exp(- np.square(dists_x[:, 1:] / dists_x[:, 1].reshape(-1, 1)))
        u_x = np.exp(- dists_x[:, 1:] / (dists_x[:, 1].reshape(-1, 1) + 1e-5 ))
        ##
        w_x = u_x / u_x.sum(axis = 1).reshape(-1, 1)
        ##
        # y_nn = np.array([[pts_y[idxs_x[i, j]] for j in range(1, m+2)] for i in range(pts_x.shape[0])])
        # # print(y_nn[:10])
        # y_hat = (y_nn * np.expand_dims(w_x, 2).repeat(m, axis=2)).sum(axis = 1)
        # # print(y_hat.shape)
        # s_y_hat = [convert_symbolic_rank(y_hat[i]) for i in range(pts_x.shape[0])]

        y_hat = np.empty((pts_y.shape[0], pts_y.shape[1]))
        for i in range(pts_y.shape[1]):
            y_hat[:,i] = (pts_y[:,i][idxs_x[:, 1:]] * w_x).sum(axis = 1)
        s_y_hat = [convert_symbolic_rank(y_hat[i]) for i in range(pts_x.shape[0])]
        ##
        # cm_rho = np.corrcoef(s_y_hat, y)[0, 1]
        cm_rho = GaussianAPMI(s_y, s_y_hat, pts_x.shape[0])
        return cm_rho
        ##
    ##
    def cm_rhos(pts, m = m, metric = metric):
        params = {'m': m, 'metric': metric}
        cm_xy = cm_rho_fun(pts[:, :mx], pts[:, mx:], **params)
        cm_yx = cm_rho_fun(pts[:, mx:], pts[:, :mx], **params)
        # print("cm_yx:",cm_yx)
        # print("cm_xy:",cm_xy)
        return cm_yx, cm_xy

    def GaussianAPMI(s_x, s_y, N):   
           
        # Stat_x, Stat_y = Counter(s_x), Counter(s_y)
        # Stat_xy = Counter(joint_rankvectors(s_x, s_y))
        # H_x, H_y, H_xy = entropy(list(Stat_x.values())), entropy(list(Stat_y.values())), entropy(list(Stat_xy.values()))
        # PMI = H_x + H_y - H_xy
        # EMI = expected_MI(list(Stat_x.values()), list(Stat_y.values()), N)
        # print(EMI)
        # APMI = (PMI - EMI) / (0.5*(H_x+H_y) - EMI + 1e-8) #if PMI>EMI else 0.0
        # APMI = max(0, APMI)
        APMI = AMI( s_x, s_y ) 
        if APMI < 0:
            APMI = 0
        return (1 - math.exp(-2*APMI))**0.5
        # return APMI


    min_T = max(m + 2, min_T)
    max_T = n
    ##
    ## Library lengths: list_T
    if plot is True or exp_fit is True:
        a = (max_T - min_T) ** (1 / (n_T - 1)) + 1e-8
        k = (max_T - min_T) / (a ** n_T)
        fun_T = lambda x: k * a ** x
        ##
        list_T = fun_T(np.arange(n_T) + 1) / fun_T(n_T) * (max_T - min_T)
        list_T = min_T + list_T.astype(int)
    else:
        list_T = np.array([min_T, max_T])
    ##
    ## Number of samples for each library length
    n_samples_in = n_samples
    n_samples = n // list_T
    if n_samples_in is not None:
        n_samples = \
            np.min((n_samples, np.repeat(n_samples_in, len(list_T))), axis = 0)

    ## Computation
    cm_vals = np.zeros((len(list_T), 2))
    for ii in range(len(list_T)):
        for jj in range(n_samples[ii]):
            if random:
                if n == list_T[ii]:
                    ind = 0
                else:
                    ind = np.random.choice(n - list_T[ii])
            else:
                ind = jj * list_T[ii]
            cm_vals[ii, :] += cm_rhos(pts[ind:ind + list_T[ii], :], m, metric)
        cm_vals[ii, :] /= n_samples[ii]
    ##
    ## If exponential curve fit (as in Monster et al [2016]):
    if exp_fit is True:
        def exp_fun(x, gamma, p_0, p_inf):
            return (p_0 - p_inf) * np.exp(- gamma * x) + p_inf
        p0 = (1e-6, 0, 0.5)
        try:
            par_yx, _ = curve_fit(exp_fun, list_T, cm_vals[:, 0], p0 = p0)
            if np.median(cm_vals[:-1, 0]) - \
                    exp_fun(list_T[0], *par_yx) < rho_tol or exp_fun(list_T[-1], *par_yx) \
                        < rho_tol or par_yx[-1]>1:
                ccm_yx = cm_vals[-1, 0]
            else:
                ccm_yx = par_yx[-1]
        except:
            ccm_yx = np.nan
        try:
            par_xy, _ = curve_fit(exp_fun, list_T, cm_vals[:, 1], p0 = p0)
            ##
            if np.median(cm_vals[:-1, 1]) - \
                    exp_fun(list_T[0], *par_xy) < rho_tol or exp_fun(list_T[-1], *par_xy) \
                        < rho_tol or par_xy[-1]>1:
                ccm_xy = cm_vals[-1, 1]
            else:
                ccm_xy = par_xy[-1]
        except:
            ccm_xy = np.nan
    else:
        if cm_vals[-1, 0] - max(cm_vals[:-1, 0]) < rho_tol or cm_vals[-1, 0] < 0:
            ccm_yx = 0
        else:
            ccm_yx = cm_vals[-1, 0]
        if cm_vals[-1, 1] - max(cm_vals[:-1, 1]) < rho_tol or cm_vals[-1, 1] < 0:
            ccm_xy = 0
        else:
            ccm_xy = cm_vals[-1, 1]
    ##
    if plot is True:
        plt.figure()
        plt.plot(list_T, cm_vals[:, 1], 'bo')
        plt.plot(list_T, cm_vals[:, 0], 'ro')
        if exp_fit:
            xx = np.arange(1, max_T)
            try:
                plt.plot(xx, exp_fun(xx, *par_xy), 'b')
                plt.plot(xx, exp_fun(xx, *par_yx), 'r')
            except:
                pass
    ##
    ## ccm_xy is the transfer entropy of y given x e.g. CCM(X|Y) / CCM_{Y->X}
    ## Note if y causally influences x, y influences dynamics of x and knowledge
    ## of shadow manifold Ax can be used to estimate y. Hence
    ## CCM(X|Y) = rho(y^, y)
    return ccm_xy, ccm_yx

import time
from joblib import Parallel, delayed
from joblib import parallel_backend
import multiprocessing

class CCM_base(object):
    def __init__(self, emb_params: tuple, min_T: int, n_samples: int, n_T=30, rho_tol=0.0, exp_fit=True) -> None:
        self.emb_dim, self.emb_lag = emb_params
        self.min_T = min_T
        self.n_samples = n_samples
        self.n_T = n_T
        self.rho_tol = rho_tol
        self.exp_fit = exp_fit
          
    def pairwise_inference(self, data1, data2):
        pass

    def network_inference(self, data, runtime_collect=True):
        """
        Infer a directed connectivity matrix from a multivariate dataset by running all
        pairwise inferences in parallel.
        This method treats each column of `data` as a variable (node) and computes a
        directed score for every ordered pair using `self.pairwise_inference`. For each
        pair (i, j) with i < j, both directions are inferred once and written into the
        output matrix. The diagonal remains zero.

        Parameters
        ----------
        data : numpy.ndarray, shape (n_samples, n_variables)
            2D array where rows are observations (e.g., time points) and columns are
            variables/nodes to be connected via pairwise inference.
        runtime_collect : bool, default True
            If True, records timing information:
            - self.cpu_time: approximate total CPU time across workers plus overhead.
            - self.wall_time: elapsed wall-clock time of the whole routine.
            Also prints the number of available CPUs and the number of parallel jobs.

        Returns
        -------
        numpy.ndarray, shape (n_variables, n_variables)
            Directed connectivity matrix C where:
            - Let pairwise_inference(x, y) -> (s_xy, s_yx).
            - Then C[y, x] = s_xy and C[x, y] = s_yx.
            Thus, rows index the source/predictor variable and columns index the
            target/response variable. The diagonal is zero.

        Notes
        -----
        - Parallelization uses joblib with the 'loky' backend and n_jobs selected as
          min(multiprocessing.cpu_count(), number_of_pairs).
        - All exceptions from `self.pairwise_inference` or the parallel backend will
          propagate; no internal retries are performed.
        """
        
        N, E = data.shape
        c_mat = np.zeros((E,E))
        if runtime_collect:
            t0_cpu, t0_wall = time.process_time(), time.time()

        def compute_pair(i, j):
            t0_cpu = time.process_time()
            if i == j:
                t1_cpu = time.process_time()
                return (i, j, 0.0, 0.0, t1_cpu - t0_cpu)
            else:
                ccm_xy, ccm_yx = self.pairwise_inference(data[:, i], data[:, j])
                t1_cpu = time.process_time()
                return (i, j, ccm_xy, ccm_yx, t1_cpu - t0_cpu)

        pairs = [(i, j) for i in range(E) for j in range(i+1, E)]
        num_cpus = multiprocessing.cpu_count()
        print(f"Number of CPUs available: {num_cpus}")
        n_jobs = min(num_cpus, len(pairs))  # Limit to number of pairs or available CPUs
        print(f"Using {n_jobs} jobs for parallel processing.")
        with parallel_backend('loky', n_jobs=n_jobs):
            results = Parallel()(
            delayed(compute_pair)(i, j) for i, j in pairs
            )
        total_cpu_time = 0.0
        for i, j, ccm_xy, ccm_yx, t_cpu in results:
            c_mat[j, i], c_mat[i, j] = ccm_xy, ccm_yx
            total_cpu_time += t_cpu
        if runtime_collect:
            t1_cpu, t1_wall = time.process_time(), time.time()
            self.cpu_time = total_cpu_time + t1_cpu - t0_cpu
            self.wall_time = t1_wall - t0_wall

        return c_mat
    

class ClassicalCCM(CCM_base):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def pairwise_inference(self, data1, data2):
        ccm_xy, ccm_yx = convergent_cross_mapping_full(data1, data2, mx=self.emb_dim, my=self.emb_dim, exp_fit=self.exp_fit,\
                                                       rho_tol=self.rho_tol, min_T=self.min_T, n_T=self.n_T, n_samples=self.n_samples)
        return ccm_xy, ccm_yx
    

class FrequencyCCM(CCM_base):
    def __init__(self, truncate_tol, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.truncate_tol = truncate_tol

    def pairwise_inference(self, data1, data2):
        ccm_xy, ccm_yx = freq_convergent_cross_mapping(data1, data2, ws=(self.emb_dim-1)*self.emb_lag, hop=self.emb_lag, exp_fit=self.exp_fit,\
                                                       rho_tol=self.rho_tol, min_T=self.min_T, n_T=self.n_T, n_samples=self.n_samples,\
                                                          truncate_tol=self.truncate_tol)    
        return ccm_xy, ccm_yx


class SymbolicCCM(CCM_base):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def pairwise_inference(self, data1, data2):
        ccm_xy, ccm_yx = symbolic_convergent_cross_mapping(data1, data2, mx=self.emb_dim, my=self.emb_dim, exp_fit=self.exp_fit,\
                                                       rho_tol=self.rho_tol, min_T=self.min_T, n_T=self.n_T, n_samples=self.n_samples)
        return ccm_xy, ccm_yx
