import scipy.io
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import torch.nn as nn

np.random.seed(951011)
def initialization(data,k):
    '''
    Initialization parameters: pi, A, phi
    :return:
    '''
    N = data.shape[0]
    pi = np.full((k),1/k)
    A = np.full((k,k), 1/k)
    d = data.shape[1]
    # initialize gaussians

    mean = np.mean(data,axis=0)
    var = np.cov(data.T)
    std = np.std(data,axis=0)

    mu = np.random.uniform(mean-std,mean+std,size=(k, d))
    sigma = np.random.uniform(var/k,var,size=(k,d, d))
    sigma = np.array([np.dot(sigma[i], sigma[i].T) for i in range(k)])  # make it into positive semi-definite

    phi =  np.array([scipy.stats.multivariate_normal.pdf(data, mu[i], sigma[i]) for i in range(k)]).T
  
    return pi,A,phi

def forward_pass(data,pi,A,phi,K):
    N = data.shape[0]  # represents time t
    alpha = np.zeros((N,K)) # alpha_i
   
    alpha[0] = phi[0]*pi # base case
    
    for t in range(1,N):
        alpha[t] = phi[t]*np.dot(A.T,alpha[t-1])
        normalizer = np.sum(alpha[t])
        alpha[t] = alpha[t]/normalizer
    return alpha

def backward_pass(data,A,phi,K):
    N = data.shape[0]
    beta = np.zeros((N, K))
    beta[-1] = np.ones(K) # initialize the base
    for t in range(N-2,-1,-1): # len(N)-1 is our base
        beta[t] = np.dot(A,beta[t+1]*phi[t+1])
        normalizer = np.sum(beta[t+1])
        beta[t] = beta[t]/normalizer
    return beta

def expectation(pi,phi,N,A,K):
    '''
    :param alpha:
    :param beta:
    :param phi:
    :param N: number of states
    :return: gamma, epsilon
    '''
    alpha = forward_pass(data,pi,A,phi,K)
    beta = backward_pass(data,A,phi,K)

    x = np.multiply(alpha,beta)
    Z_gamma = np.sum(x,axis=1)
    gamma = x / Z_gamma[:, None]

    y = phi*beta
    Z_epsilon = np.zeros(N-1)

    # cross-time statistics
    for t in range(N-1):
        Z_epsilon[t] = np.dot(alpha[t],np.dot(A,y[t+1]))
    all_time_epsilon = np.zeros((N-1,K,K))
    for t in range(N-1):
        all_time_epsilon[t] = (A*alpha[t][:,np.newaxis])*y[t+1][np.newaxis,:]
        all_time_epsilon[t] = all_time_epsilon[t]/Z_epsilon[t]
    epsilon = np.sum(all_time_epsilon,axis = 0) # cross-time statisics up to the time N
    return gamma,epsilon,alpha,beta,all_time_epsilon

def maximization(gamma,epsilon,data,K):
    '''
    :param gamma: NxK
    :param epsilon: KxK
    :param gaussians: Kx1: each is a gaussian dist'n
    :return: updated parameters for all pi,phi,A,gaussian parameters
    '''
    d = data.shape[1]
    numer = np.sum(gamma, axis=0)  
    denom = np.sum(numer)
    pi = numer / denom
    z = np.sum(gamma,axis=0)

    # update gaussian parameters
    mu = np.sum(np.einsum('ij,ik->kij', data, gamma),axis=1)/z[:,np.newaxis] # KxD
    sigma = np.zeros((K,d,d))
    for k in range(K):
        k_cov= np.dot(((data - mu[k]) * gamma[:,k][:, np.newaxis]).T, (data-mu[k]))
        sigma[k] = k_cov/(z[k])
        sigma[k] += np.identity(d)*0.1
    gaussian = [mu,sigma]
    phi =  np.array([scipy.stats.multivariate_normal.pdf(data, mu[i], sigma[i]) for i in range(K)]).T
    A = epsilon/np.sum(epsilon,axis=1)[:,np.newaxis]
    return pi,phi,A,gaussian

def EM(data, k, iters=100):
    '''

    :param data: input data Nxd
    :param k: possible value of each hidden state
    :return:
    '''
    pi,A,phi = initialization(data,k)
    

    N = data.shape[0]
    for i in range(iters):
        gamma, epsilon,alpha,beta,all_time_epsilon = expectation(pi,phi,N,A,k)
        pi,phi,A,gaussian = maximization(gamma,epsilon,data,k)
        #print("Finish {}".format(i))
    return gamma, epsilon, pi,phi,A,gaussian,all_time_epsilon


def contour(data,mu,sigma):
    z = scipy.stats.multivariate_normal.pdf(data,mu,sigma)
    return z

def visualization(data, gaussian,k,lower,upper):

    X= np.linspace(lower, upper, 100)
    Y = np.linspace(lower, upper, 100)
    X, Y = np.meshgrid(X, Y)
    for i in range(len(gaussian[0])):
        Z = np.array([contour(np.array(point).reshape(1,2),gaussian[0][i],gaussian[1][i])
                      for point in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
        plt.contour(X, Y, Z)

    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel(r'$x_1$')    
    plt.ylabel(r'$x_2$')
    plt.title('HMM')

    plt.show()



def BIC(N,R,pi,gamma,A,phi,epsilon):
	first_term_likelihood = np.sum(np.dot(np.log(pi),gamma[0]))
	
	second_term_likelihood = np.sum(np.log(np.sum(np.multiply(phi,gamma),axis=1)))
	third_term_likelihood = np.sum(np.multiply(np.log(A),epsilon))
	L = first_term_likelihood+second_term_likelihood+third_term_likelihood
	BIC = np.log(N)*R-2*L
	return BIC


def viterbi_deco(X, pi, A, emission_prob,S):
    '''
    Computational Time Complexity: O(TS^2); space complexity O(TS)
    Choose state path with largest probability to maximize expected num of correct states
    Most probable sequence of states
    May not be the best path with highest likelihood of generating data
    z = argmax_z p(z|x)
    :param: X: a sequence of observations; value for this assignment; index of item for test purposes
    :param: pi: initial prob
    :param: A: transition matrix
    :param: emission_prob; uncomment gaussian for THIS assignment ONLY gaussian = [mu,sigma] mu[i] is i-th hidden states parameters
    :return:
    '''
    T = X.shape[0] # num of observations
    S = emission_prob.shape[1] # num of states

    #S = emission_prob[0].shape[0]
    # let compute_array_st to be best probability at state s at time t
    compute_array = np.zeros((S,T)) # matrix each row is state and each column is time
    memo_array = np.zeros((S,T))
    # initialization; at time=0;
    for s in range(S):
        #prob_x_s = emission_prob[X[0]][s]
        prob_x_s = scipy.stats.multivariate_normal.pdf(X[0], emission_prob[0][s], emission_prob[1][s])
        compute_array[s][0] = pi[s]*prob_x_s

    # induction: the best path to state s at time t, depends on the best path to each possible previous stats
    #               and the transition to s A[:,j]
    for t in range(1,T):
        for s in range(S):
            #prob_x_s = emission_prob[X[t]][s]
            prob_x_s = scipy.stats.multivariate_normal.pdf(X[t], emission_prob[0][s], emission_prob[1][s])
            compute_array[s][t] = np.max((compute_array[:,t-1]*A[:,s])*prob_x_s)
            memo_array[s][t] = np.argmax((compute_array[:,t-1]*A[:,s])*prob_x_s)

    # back tracing for path
    path = []
    k = np.argmax(compute_array[:,T-1])
    path.append(k)
    for t in range(T-1,0,-1):
        path.append(memo_array[k][t])
        k = int(memo_array[k][t])
    # need to reverse the order
    return path[::-1]

def test_viterbi_deco(S):
    S = np.array([0,1,2]) # 0 is disgusted, 1 is happy, 2 is sad
    A = np.array([[0.4,0.5,0.1]
                 ,[0,0.8,0.2]
                 ,[0,0,1]])
    emission_prob = np.array([[0.1,0.05,0.05,0.6,0.05,0.1,0.05],[0.3,0,0,0.2,0.05,0.05,0.4],[0.25,0.25,0.05,0.3,0.05,0.09,0.01]]).T
    pi = np.array([0.8,0.2,0.0])
    X = np.array([0,3,6])
    return viterbi_deco(X, pi, A, emission_prob,S)

if __name__ == "__main__":
    pass

    # data = scipy.io.loadmat('HMM-data.mat')
    # data = data['X'][:1000]
    #
    #
    # # second parameter is the num of possiblt state; choose one from 2,3,4,5
    #
    # gamma,epsilon, pi,phi,A,gaussian,all_time_epsilon = EM(data, 2, iters=100)
    #upper = np.max(data)
    #lower = np.min(data)

    #visualization(data, gaussian,2,lower,upper)

    ######
##    BIC_result = []
##    for i in [2,3,4,5]:
##        gamma,epsilon, pi,phi,A,gaussian,all_time_epsilon = EM(data, i, iters=100)
##        R = 2*i + (i)*2*(2+1)/2 + i*(i-1)
##        print(R)
##        res = BIC(data.shape[0],R,pi,gamma,A,phi,epsilon) # R needed to be changed here
##        BIC_result.append(res)
##        print("The BIC for {} num of values of k is {}".format(i,res))
        
    
    
