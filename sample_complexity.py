#sample complexity supervised

from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import cProfile

def perceptron(X,y,x_test, gamma = 0.001):
    '''returns the perceptron prediction. works offline'''
    
    w = np.zeros(X.shape[1])
    error = float('inf')
    while error > gamma:
        error = 0
        for i in range(X.shape[0]):
            x = X[i]
            pred = 2 * (np.dot(w,x) > 0) - 1
            if pred != y[i]:
                error += 1
                w += y[i] * x
        error /= X.shape[0]
    #converged
    return 2 * (np.matmul(x_test, w) > 0) - 1
    
def winnow(X,y,x_test, gamma = 0.001):
    '''returns the winnow prediction. works offline.'''
    
    X = (X + 1)/2
    y = (y + 1)/2
    x_test = (x_test + 1)/2
    n = X.shape[1]
    w = np.ones(n)
    error = float('inf')
    while error > gamma:
        error = 0
        for i in range(X.shape[0]):
            x = X[i]
            pred = (np.dot(w,x) >= n)
            if pred != y[i]:
                error += 1
                w *= 2 ** ((y[i] - pred) * x)
        error /= X.shape[0]
    #converged
    return 2 * (np.matmul(x_test, w) >= n) - 1
    
    
def least_squares(X,y,x_test):
    '''returns the least-squares prediction'''
    
    w = np.matmul(np.linalg.pinv(X),y)
    pred = np.matmul(x_test,w) 
    return 2 * (pred > 0) - 1
    
def one_nn(X,y,x_test):
    '''returns the one-nearest neighhbour prediction on x_test
    given training data X (examples are rows) with labels y.
    x_test is a 2d array'''
    
    x_test = np.swapaxes(np.expand_dims(x_test,2),0,2)
    X = np.expand_dims(X,2)
    closest = np.argmin(np.count_nonzero(X != x_test, axis = 1), axis = 0)
    return y[closest]

X = np.array([[1, -1, 1],
              [1, 1, -1],
              [1, -1, 1],
              [-1, 1, 1]])

y = np.array([[1,1,1,-1]]).T

x_test = np.array([[1,1,1],
                   [1,1,-1]])

#print one_nn(X,y,x_test)
#print least_squares(X,y,x_test)
#print perceptron(X,y,x_test)
#print winnow(X,y,x_test)

def sample_complexity(algo, max_n):
    '''creates graph of sample complexity up to dimension max_n,
    for classifier algo'''
    
    ms = np.zeros(max_n)
    for n in range(1,max_n + 1):
        print "n =", n
        m = 1
        while True:
            #print "m =", m
            num_samples = m
            error = 0
            for i in range(num_samples):
                #generate sample of size m and dim n
                X = 2 * np.random.multinomial(1,\
                                pvals=[0.5,0.5],size = m * n).argmax(axis=1) - 1
                X = np.array(X).reshape((m,n))
                y = X[:,0]
                
                #generate test inputs of dim n
                test_size = n
                x_test = 2 * np.random.multinomial(1,\
                                pvals=[0.5,0.5],size=test_size * n).argmax(axis=1) - 1
                x_test = np.array(x_test).reshape((test_size, n))
                y_test = x_test[:,0]

                #estimate generalisation errors                
                pred = algo(X,y,x_test)
                error += np.count_nonzero(pred != y_test)/test_size
                
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

sample_complexity(least_squares,100)
sample_complexity(perceptron, 100)
sample_complexity(winnow, 150)
sample_complexity(one_nn,16)


# pr = cProfile.Profile()
# pr.enable()
# sample_complexity(one_nn,13)
# pr.disable()
# pr.print_stats(sort='time')
