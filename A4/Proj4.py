
import scipy.io as sio


import numpy as np  

import os

import matplotlib.pyplot as plt  


from scipy.special import logsumexp

from numpy.polynomial.polynomial import polyvander



################################ Data #########################################################

os.chdir('/Users/Lai/Dropbox/CS/sherrie/A4')
data = sio.loadmat('data4.mat')
dataX = np.array(data['X'])
dataY = np.array(data['Y'])[:, 0]
data_labels = np.array(data['Z'])[:, 0]  # 0 missing labels, 1 is class 1, 2 is class 2
np.random.seed(12)

N = dataX.shape[0]
D = dataX.shape[1]
C = 3
P = 4

############################### Helper Funcs #################################################


def log_factorial_approximation(Y):
    result = np.zeros(N)
    for j in range(N):
        for i in range(Y[j][0], 0, -1):
            result[j] += np.log(i)
    return result.reshape(-1,1)

def log_Poisson(log_lambda,Y):
    '''
    Keep in the log space for numerical stability
    :param log_lambda:
    :param Y:
    :return:
    '''
    Y = Y.reshape(-1,1)
    result = -log_lambda + Y * log_lambda - log_factorial_approximation(Y)
    return result


def basis_func(x,p):
    X = np.ones((N,1))
    for i in range(0, 2):
      for j in range(1, p + 1):
        tmp = np.reshape(x[:,i]**j, (N,1))
        X = np.concatenate([X, tmp], axis = 1)
    return X
    #return polyvander(x, deg=P).reshape(N, D*(P+1))

def log_lambda_update(X,beta):
    return np.dot(X,beta)

def taylor_exp(x):
    return 1+x+x**2/2

def objective_func(Z, Y, log_lambda):

    log_f = log_Poisson(log_lambda,Y)

    return np.mean(Z*log_f + Z)



######################################## Algorithm #########################################
  
def init_params(X,Y):

    Z = np.zeros((N,C))
    for i in range(N):
        if data_labels[i] == 1:
            Z[i][1] = 1
        elif dataY[i] == 2:
            Z[i][2] = 1
        else:
            Z[i,:] = 1/C

    indx = np.where(dataY != 0)[0]
    vdata_labels = data_labels[indx]
    vdataY = dataY[indx]

    log_lambda = np.zeros((N, C))
    for c in range(C):
      log_lambda[:, c] = np.mean(np.log(vdataY[np.where(vdata_labels == c)[0]], dtype='f'))
    beta,gamma,log_lambda,W = Gradient_Ascent(Y.reshape(-1, 1), log_lambda, X, Z, 100,0.2)
    return X, beta, Z

def Expectation(alpha,log_lambda,Y):
    '''
    :return: the responsibility
    '''
    log_poisson_dist = log_Poisson(log_lambda,Y)
    log_proportion = np.log(alpha)
    nume = log_proportion + log_poisson_dist
    deno = logsumexp(nume,axis=1)
    return np.exp(nume - deno[:, None])

def Gradient_Ascent(Y, log_lambda, X, Z, iters,learning_rate):
    # Gradient Ascend to approximate optimal lambda
    temp = Z * taylor_exp(log_lambda)
    W = np.zeros((C, N, N))
    for c in range(C):
        W[c] = np.diag(temp[:, c])

    for t in range(iters):
        gamma = np.zeros((N, C))
        for i in range(0, C):
            gamma = log_lambda + (Y.reshape(-1, 1) - taylor_exp(log_lambda)) / taylor_exp(log_lambda) * learning_rate

            # beta = np.zeros((D * (P + 1), C))
        beta = np.zeros((D * P + 1, C))
        for i in range(0, C):
            U, Omega, V = np.linalg.svd(np.matmul(W[i] ** 0.5, X[:]), full_matrices=0)
            beta[:, i] = np.matmul(V * np.linalg.inv(np.diag(Omega)), U.T.dot(np.matmul(W[i] ** 0.5, gamma[:, i].T)))

        log_lambda = np.zeros((N, C))
        # update log(lambda)
        if beta[0].size != 1:
            log_lambda = log_lambda_update(X, beta)
        else:
            for j in range(0, C):
                log_lambda[:, j] = beta[j]
    return beta, gamma, log_lambda, W

def Optimization(x, y, iters,learning_rate):
    # initial guess
    X, beta, Z = init_params(x,y)
    for i in range(iters):
        log_lambda = log_lambda_update(X, beta)
        alpha = np.sum(Z, axis=0) / N
        beta, gamma, log_lambda, W = Gradient_Ascent(y.reshape(-1, 1), log_lambda, X, Z, 100,learning_rate)
        Z = Expectation(alpha, log_lambda, y)
        if i%20 == 0:
            print("EM {}th Iteration".format(i))

    PlotRsd(dataX, X, data_labels, Z, beta, log_lambda, P, C)
    return beta, Z, log_lambda, alpha

################################# Simulation ##################################################

























################################## Visualization ##############################################


def PlotRsd(x, X, z, Z, beta, log_lambda, degree, K):
    #beta = beta[::-1,:]
    beta = [beta[:,0],beta[:,1],beta[:,2]]
    # find the dimension and length of the data 
    N = z.size
    d = x.shape[1]
    for i in range(0, K):
        print ("-------Estimation for class:", i+1, "------- order:", degree)
        for j in range(d):  
            xx = np.linspace(np.min(x[:,j], axis = 0)*1.1, np.max(x[:,j], axis = 0)*1.1, 10000)
            XX = np.reshape(xx, (10000,1))
            for k in range(2, degree + 1):
                tmp = np.reshape(xx**k, (10000,1))
                XX = np.concatenate([XX, tmp], axis = 1)   
            I = (Z[:,i] == np.max(Z, axis=1))
            #plot_rsd = np.log(y[I]) - beta[i][0]
            plot_rsd = log_lambda[I,i] - beta[i][0]
            for k in range(d):
                plot_rsd = plot_rsd - (j != k) * np.dot(X[I][:,k*degree+1:(k+1)*degree+1], beta[i][k*degree+1:(k+1)*degree+1])          
            plt.subplot(1,3,j+1)
            I1 = I & (z != 0)
            plt.scatter(x[:,j][I1], plot_rsd[z[I]!=0].T, s = 20, label = "has label")           
            I2 = I & (z == 0)
            plt.scatter(x[:,j][I2], plot_rsd[z[I]==0].T, s = 10, label = "missing label", color = "yellow")
            if I.sum() > 0: plt.plot(xx, np.dot(XX, beta[i][j*degree+1:(j+1)*degree+1]), 'r')
            plt.xlabel("x" + str(j+1))
            plt.ylabel("residual")
            plt.legend()
        print (I1.sum())
        print (I2.sum())
        plt.subplot(1,3,3)
        plt.scatter(log_lambda[I1, i], np.dot(X[I1], beta[i]), s = 20, label = "has label")
        plt.scatter(log_lambda[I2, i], np.dot(X[I2], beta[i]), s = 10, label = "missing label", color = "yellow")
        if I.sum() > 0: plt.plot([min(log_lambda[I,i]), max(log_lambda[I,i])], [min(log_lambda[I,i]), max(log_lambda[I,i])], 'r')  
        plt.xlabel("actual log_lambda score")
        plt.ylabel("predict log_lambda score")
        plt.legend()
        plt.show()


X = basis_func(dataX,P)
beta, Z, log_lambda,alpha = Optimization(X, dataY, data_labels, 40,0.2)

