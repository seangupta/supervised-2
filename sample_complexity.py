#sample complexity supervised

from __future__ import division
from matplotlib import pyplot as plt
import numpy as np

def perceptron():
    pass
    
def winnow():
    pass
    
def least_squares():
    pass
    
def one_nn(X,y,x_test):
    '''returns the one-nearest neighhbour prediction on x_test
    given training data X (examples are rows) with labels y.
    x_test is a 2d array'''
    
    x_test = np.swapaxes(np.expand_dims(x_test,2),0,2)
    x_test = np.tile(x_test,(X.shape[0],1,1))
    X = np.tile(np.expand_dims(X,2),(1,1,x_test.shape[2]))
    dists = np.sum(X != x_test,axis = 1)
    closest = np.argmin(dists,axis=0)
    return y[closest]

X = np.array([[1, -1, 1],
              [1, 1, -1],
              [1, -1, 1],
              [-1, 1, 1]])

y = np.array([[1,1,1,-1]]).T

x_test = np.array([[1,1,1],
                   [1,1,-1]])

#print one_nn(X,y,x_test)

def sample_complexity(algo, max_n):
    '''creates graph of sample complexity up to dimension max_n,
    for classifier algo'''
    
    ms = np.zeros(max_n)
    for n in range(1,max_n + 1):
        print "n =", n
        m = 1
        while True:
            #print "m =", m
            num_samples = m ** 2
            error = 0
            for i in range(num_samples):
                #generate sample of size m and dim n
                X = 2 * np.random.multinomial(1,\
                                pvals=[0.5,0.5],size = m * n).argmax(axis=1) - 1
                X = np.array(X).reshape((m,n))
                y = X[:,0]
                
                #generate k test inputs of dim n
                test_size = n ** 2
                x_test = 2 * np.random.multinomial(1,\
                                pvals=[0.5,0.5],size=test_size * n).argmax(axis=1) - 1
                x_test = np.array(x_test).reshape((test_size, n))
                y_test = x_test[:,0]
        
                #estimate generalisation errors
                pred = algo(X,y,x_test)
                error += np.mean(pred != y_test)
            error /= num_samples
            
            #check whether this is less than 0.1
            if error <= 0.1:
                ms[n - 1] = m
                break
            m += 1
            
    plt.figure()
    plt.plot(range(1,max_n + 1),ms)
    plt.xlabel('dimension n')
    plt.ylabel('sample complexity m')
    plt.title('sample complexity')
    plt.show()
        
sample_complexity(one_nn,10)