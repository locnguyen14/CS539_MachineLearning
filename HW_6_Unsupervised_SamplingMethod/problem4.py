#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 4: LDA (Latent Dirichlet Allocation) using Gibbs sampling method
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.

    Notations:
            ---------- dimensions ------------------------
            m: the number of data documents, an integer scalar.
            n: the number of words in each document, an integer scalar.
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            k: the number of topics, an integer scalar
            ---------- model parameters ----------------------
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            beta: the parameters of word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            -----------------------------------------------
'''

#------------------------------------------------------------------------------
#  Sampling Methods (for Bayesian Networks)
#------------------------------------------------------------------------------

'''
    Let's first practise with a simpler case: a network with only 3 random variables.

        X1 --->  X2  --->  X3

    Suppose we are sampling from the above Bayesian network.
 
'''

#--------------------------
def prior_sampling(n,PX1,PX2,PX3):
    '''
        Use prior sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
    '''
    #########################################
    ## INSERT YOUR CODE HERE





    #########################################
    return S 



'''
    Now let's assume we observe the value of X2 (evidence).

        X1 --->  X2(observed)  --->  X3
    Use different sampling methods to sample data from the above Bayesian network.
'''
#--------------------------
def rejection_sampling(n,PX1,PX2,PX3, ev):
    '''
        Use rejection sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
            ev: the observed value of X2, an integer scalar of value 0 or 1.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
    '''
    #########################################
    ## INSERT YOUR CODE HERE








    #########################################
    return S 

#--------------------------
def importance_sampling(n,PX1,PX2,PX3, ev):
    '''
        Use importance (likelihood) sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
            ev: the observed value of X2, an integer scalar of value 0 or 1.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
            w: the weights of samples, a float vector of length n.
                w[i] denotes the weight(likelihood) of the i-th sample in S.
    '''
    #########################################
    ## INSERT YOUR CODE HERE






    #########################################
    return S, w 


#------------------------------------------------------------------------------
# Gibbs Sampling
#------------------------------------------------------------------------------
'''
    Gibbs sampling: Let's switch to the following network (X3 is observed).
        X1 --->  X2  --->  X3 (observed)
'''

#--------------------------
def sample_X1(X2,X3,PX1,PX2,PX3):
    '''
        re-sample the value of X1 given the values of X2 and X3 
        Input:
            X2: the current value of X2 , an integer of value 0 or 1.
            X3: the current value of X3 , an integer of value 0 or 1.
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            X1: the generative sample of X1, an integer scalar of value 0 or 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE



    #########################################
    return X1 

#--------------------------
def sample_X2(X1,X3,PX1,PX2,PX3):
    '''
        re-sample the value of X2 given the values of X1 and X3 
        Input:
            X1: the current value of X1 , an integer of value 0 or 1.
            X3: the current value of X3 , an integer of value 0 or 1.
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            X2: the generative sample of X2, an integer scalar of value 0 or 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE



    #########################################
    return X2 

#--------------------------
def gibbs_sampling(n,X1,X2,X3,PX1,PX2,PX3):
    '''
        Use Gibbs sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            X1: the initial values of X1, an integer scalar.
            X2: the initial values of X2, an integer scalar.
            X3: the observed values of X3, an integer scalar.
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
    '''
    #########################################
    ## INSERT YOUR CODE HERE







    #########################################
    return S 



#------------------------------------------------------------------------------
#  Gibbs Sampling Methods for LDA
#------------------------------------------------------------------------------

#--------------------------
def resample_z(w,d,z,nz,nzw,ndz,alpha=1.,eta=1.):
    '''
        Use Gibbs sampling to re-sample the topic (z) of one word in a document, and update the statistics (nz, nzw,nmz) accordingly.
        Input:
            w:  the index of the current word (in the vocabulary), an integer scalar. 
                if w = i, it means that the current word in the document is the i-th word in the vocabulary. 
            d:  the id of the current document, an integer scalar (ranging from 0 to m)
            z:  the current topic assigned the word, an integer scalar (ranging from 0 to k-1).
            nz:  the frequency counts for each topic, an integer vector of length k.
                The (i)-th entry is the number of times topic i is assigned in the corpus
            nzw:  the word frequence count for each topic , an integer matrix of shape k by p.
                The (i,j) entry of is the number of times the j-th word in the vocabulary is assigned to topic i.
            ndz:  the topic frequence count for each document, an integer matrix of shape m by k.
                The (i,j) entry is the number of words in document i assigned to topic j
            alpha: the parameter for topic prior (Dirichlet distribution), a float scalar.
            eta: the parameter for word prior (Dirichlet distribution), a float scalar.
        Output:
            z: the resampled topic of the current word
            p: the vector of probability of generating each topic for the current word, a float vector of length k.
                p[i] is the probability of the current word to be assigned to the i-th topic.
    '''
    #########################################
    ## INSERT YOUR CODE HERE


    # remove the current word from the statistics in nz,nzw,nmz



    # compute the probability of generating each topic for the current word.


    # sample z according to the probability


    # update statistics (nz, nzw,nmz) with the newly sampled topic



    #########################################
    return z, p 

#--------------------------
def resample_Z(W,Z,nz,nzw,ndz,alpha=1.,eta=1.):
    '''
        Use Gibbs sampling to re-sample the topics (Z) of all words in all documents (for one pass), and update the statistics (nz, nzw,nmz) accordingly.
        Input:
            W:  the document matrix, an integer numpy matrix of shape m by n. 
                W[i,j] represents the index (in the vocabulary) of the j-th word in the i-th document.
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            Z:  the current topics assigned to all words in all documents, an integer matrix of shape m by n (value ranging from 0 to k-1).
            nz:  the frequency counts for each topic, an integer vector of length k.
                The (i)-th entry is the number of times topic i is assigned in the corpus
            nzw:  the word frequence count for each topic , an integer matrix of shape k by p.
                The (i,j) entry of is the number of times the j-th word in the vocabulary is assigned to topic i.
            ndz:  the topic frequence count for each document, an integer matrix of shape m by k.
                The (i,j) entry is the number of words in document i assigned to topic j
            alpha: the parameter for topic prior (Dirichlet distribution), a float scalar.
            eta: the parameter for word prior (Dirichlet distribution), a float scalar.
        Output:
            Z: the resampled topics of all words in all documents
    '''
    #########################################
    ## INSERT YOUR CODE HERE




    #########################################
    return Z




#--------------------------
def gibbs_sampling_LDA(W,k,p,Z,alpha=1., eta=1.,n_samples=10, n_burnin=100, sample_rate=10):
    '''
        Use Gibbs sampling to generate a collection of samples of z (topic of each word) from LDA model.
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                W[i,j] represents the index (in the vocabulary) of the j-th word in the i-th document.
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            k: the number of topics, an integer scalar.
            p: the number of words in the vocabulary, an integer scalar.
            Z:  the initial topics assigned to all words in all documents, an integer matrix of shape m by n (value ranging from 0 to k-1).
            n_samples: the number of samples to be drawn, an integer scalar.
            n_burnin: the number of samples to skip at the begining of the gibbs sampling, an integer scalar.
            sample_rate: for how many passes to select one sample, an integer scalaer. 
                         When the iteration number i % sample_rate = 0, take the sample for output.
                        For example, if sample_rate=3, n_burnin = 5, n_samples=4, the samples chosen (highlighted in parenthesis) will be on pass numbers: 0,1,2,3,4,5,(6),7,8,(9),10,11,(12),13,14,(15)
        Output:
            S_nz: the sum of nz counts in all the samples, an integer vector of length k.
            S_nzw: the sum of nzw counts in all the samples, an integer matrix of shape k by p.
            S_ndz: the sum of ndz counts in all the samples, an integer matrix of shape m by k.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # initialize nz, nzw, ndz based upon W and Z









    # burnin 






    # collect one sample every sample_rate iterations






    #########################################
    return S_nz,S_nzw,S_ndz 

#--------------------------
def compute_theta(ndz, alpha=1.):
    '''
        compute theta based upon the statistics of the samples from Gibbs sampling.
        Input:
            ndz:  the topic frequence count for each document, an integer matrix of shape m by k.
                The (i,j) entry is the number of words in document i that are assigned to topic j
            alpha: the parameter for topic prior (Dirichlet distribution), a float scalar.
        Output:
            phi: the updated estimation of parameters for topic mixture of each document, a numpy float matrix of shape m by k. 
                Each element theta[i] represents the vector of topic mixture in the i-th document. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE





    #########################################
    return theta 

#--------------------------
def compute_beta(nzw,eta=1.):
    '''
        compute beta based upon the statistics of the samples from Gibbs sampling. 
        Input:
            nzw:  the word frequence count for each topic , an integer matrix of shape k by p.
                The (i,j) entry of is the number of times the j-th word in the vocabulary is assigned to topic i.
            eta: the parameter for word prior (Dirichlet distribution), a float scalar.
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE






    #########################################
    return beta 


#--------------------------
def LDA(W,k=3,p=100,alpha=.1,eta=1.,n_samples=10, n_burnin=100, sample_rate=10):
    '''
        Variational EM algorithm for LDA. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                W[i,j] represents the index (in the vocabulary) of the j-th word in the i-th document.
                Here m is the number of text documents, an integer scalar.
            k: the number of topics, an integer scalar
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            alpha: the alpha parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
            eta: the eta parameter of the Dirichlet distribution for generating word distribution for each topic, a float scalar (eta>0).
            n_samples: the number of samples to be drawn in Gibbs sampling, an integer scalar.
            n_burnin: the number of samples to skip at the begining of the Gibbs sampling, an integer scalar.
            sample_rate: sampling rate for Gibbs sampling, an integer scalaer. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # randomly initialize Z 

    # Gibbs sampling

    # compute theta

    # compute beta

    #########################################
    return beta,theta




