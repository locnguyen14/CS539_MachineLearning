import math
import numpy as np
from problem3 import Bag
#-------------------------------------------------------------------------
'''
    Problem 4: Random Forest (on continous attributes)
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''


#-----------------------------------------------
class RF(Bag):
    '''
        Random Forest (with contineous attributes)
        Hint: Random Forest is a subclass of Bagging class in problem3. So you can reuse and overwrite the code in problem 3.
    '''

    #--------------------------
    def best_attribute(self, X, Y):
        '''
            Find the best attribute to split the node. (Overwritting the best_attribute function in the parent class: DT).
            The attributes have continous values (int/float).
            Here only a random sample of m features are considered. m = floor(sqrt(p)).
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        # INSERT YOUR CODE HERE
        # Find the subset of attribute and the corresponding values
        p = X.shape[0]
        m = int(np.floor(np.sqrt(p)))
        subset_index = np.random.choice(p, m)

        # Find the best attribute for that particular subset
        i, th = Bag.best_attribute(self, X[subset_index], Y)
        if (th == -np.inf):
            return RF.best_attribute(self, X, Y)
        i = subset_index[i]

        #########################################
        return i, th

    #--------------------------
    @staticmethod
    def load_dataset(filename='data4.csv'):
        '''
            Load dataset 4 from the CSV file:data3.csv.
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a float scalar.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is an integer.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        data = np.genfromtxt(filename, delimiter=",", skip_header=1, dtype='str')
        X = np.delete(data, 0, axis=1).T.astype(float)
        Y = data[:, 0].astype(int)
        #########################################
        return X, Y
