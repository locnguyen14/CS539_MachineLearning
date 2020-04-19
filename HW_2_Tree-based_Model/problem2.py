import math
import numpy as np
from problem1 import Tree
#-------------------------------------------------------------------------
'''
    Problem 2: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#--------------------------


class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs:
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''

    def __init__(self, X=None, Y=None, i=None, th=None, C1=None, C2=None, isleaf=False, p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th
        self.C1 = C1
        self.C2 = C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X, Y):
        '''
            Find all possible cutting points in the continous attribute of X.
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        # Sort the X values
        X_sorted = np.sort(np.unique(X))

        # Calculate cutting points between two consecutive points
        cp_list = []
        for i in range(np.size(X_sorted) - 1):
            label_array = np.concatenate((Y[X == X_sorted[i]], Y[X == X_sorted[i + 1]]), axis=None)

        # Check whether the class label array is uniform. If not, we make the cut
            if (not Tree.stop1(label_array)):
                cp_list.append((X_sorted[i] + X_sorted[i + 1]) / 2.0)

        # Checking if there's no candidate for cutting points
        if (len(cp_list) == 0):
            return -np.inf

        #########################################
        return np.array(cp_list)

    #--------------------------
    @staticmethod
    def best_threshold(X, Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar.
                g: the information gain by using the best threhold, a float scalar.
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        # INSERT YOUR CODE HERE

        # Find a list of possible cutting points
        threshold_potential = DT.cutting_points(X, Y)

        # Check for input of threhodl potential
        if np.all(threshold_potential == -np.inf):
            return -float('inf'), -1

        # Getting infomation_gain list
        info_list = []
        for threshold in np.nditer(threshold_potential):
            info_list.append(Tree.information_gain(Y, X >= threshold))

        # Getting the best threshold and information gain using best threshold
        g = max(info_list)
        th = threshold_potential[np.argmax(info_list)]
        #########################################
        return th, g

    #--------------------------
    def best_attribute(self, X, Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
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
        # Get an array of best threshold and information gain for each attribute
        threshold_gain = np.array(DT.best_threshold(X[0, :], Y))
        for i in range(1, X.shape[0]):
            threshold_gain_i = np.array([DT.best_threshold(X[i, :], Y)])
            threshold_gain = np.vstack((threshold_gain, threshold_gain_i))

        # Choose the best attribute and its corresponding threshold
        i = np.argmax(threshold_gain[:, 1])
        th = threshold_gain[i][0]

        #########################################
        return i, th
    #--------------------------

    @staticmethod
    def split(X, Y, i, th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y
            (3) build children nodes by assigning a submatrix of X and Y to each node

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        # INSERT YOUR CODE HERE
        C1 = Node(X=X.T[np.where(X[i] < th)].T, Y=Y[X[i] < th])
        C2 = Node(X=X.T[np.where(X[i] >= th)].T, Y=Y[X[i] >= th])
        #########################################
        return C1, C2

    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        # INSERT YOUR CODE HERE
        t.p = Tree.most_common(t.Y)

        # if Condition 1 or 2 holds, stop recursion
        if (Tree.stop1(t.Y) or (Tree.stop2(t.X))):
            t.isleaf = True
            return

        # find the best attribute to split
        t.i, t.th = DT.best_attribute(self, t.X, t.Y)

        # recursively build subtree on each child node
        t.C1, t.C2 = DT.split(t.X, t.Y, t.i, t.th)
        self.build_tree(t.C1)
        self.build_tree(t.C2)
        return
        #########################################

    #--------------------------
    @staticmethod
    def inference(t, x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively.
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        while (t.isleaf is False):
            if(x[t.i] < t.th):
                t = t.C1
            else:
                t = t.C2
        # predict label, if the current node is a leaf node
        y = t.p
        #########################################
        return y

    #--------------------------
    @staticmethod
    def predict(t, X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset.
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        Y = np.empty(0)
        for instance in X.T:
            Y = np.append(Y, DT.inference(t, instance))
        #########################################
        return Y.T

    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        t = Node(X=X, Y=Y)
        DT.build_tree(self, t)
        #########################################
        return t

    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv.
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
                   Each element is a string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        data = np.genfromtxt(filename, delimiter=",", skip_header=1, dtype='str')
        X = np.delete(data, 0, axis=1).T.astype(float)
        Y = data[:, 0]
        #########################################
        return X, Y
