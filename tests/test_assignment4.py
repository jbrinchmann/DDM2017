# Simple test script

import unittest
from Problemsets.Problem_set_4.solutions_real import *
import sqlite3 as lite
import os

class assignment4_test(unittest.TestCase):
    def setUp(self):
        self.cwd=os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)

    def assertAlmostTrue(self,x,y):
        try:
            np.testing.assert_almost_equal(x,y)
        except:
            return False
        return True

    def test_read_csv(self):
        data=load_data_from_csv(self.cwd+'/Datasets/x-vs-y-for-PCA.csv') 
        self.assertTrue(data.keys()[1]=='x')
        self.assertTrue(len(data['x'])==40)

    def test_standardize_dim(self):
        self.assertAlmostTrue(np.array([-0.70710678, -0.70710678,  1.41421356]),standardize([1,1,2]))
        self.assertAlmostTrue(standardize([1,2,3,4,5]),np.array([-1.41421356, -0.70710678,  0.,  0.70710678,  1.41421356]))
        self.assertAlmostTrue(standardize([0,0,0,0]),np.array([0,0,0,0]))

    def test_stand_data(self):
        a=standardize_data(load_data_from_csv(self.cwd+'/Datasets/x-vs-y-for-PCA.csv'))
        self.assertTrue(np.std(a['x'])>0.99999)
        self.assertTrue(np.std(a['y'])>0.99999)
        self.assertTrue(np.mean(a['x'])<0.00001)
        self.assertTrue(np.mean(a['y'])<0.00001)       

    def test_covar_matrix(self):
       a=standardize_data(load_data_from_csv('Datasets/x-vs-y-for-PCA.csv'))
       self.assertAlmostTrue(get_cov_data(a),np.array([[ 1.02564103,  0.99718328 ],
                   [ 0.99718328,  1.02564103 ]]))

    def test_covar_vectors(self):
        a=standardize_data(load_data_from_csv(self.cwd+'/Datasets/x-vs-y-for-PCA.csv'))
        c=get_cov_data(a)
        self.assertAlmostTrue(get_covar_eigenvectors(c)['lam'],
                np.array([ 0.02845774,  2.02282431 ]))
        self.assertAlmostTrue(get_covar_eigenvectors(c)['v'],
                                 np.array([[-0.70710678, -0.70710678],
                                           [ 0.70710678, -0.70710678 ]]))
    def test_eigen_components(self):
        a=standardize_data(load_data_from_csv(self.cwd+'/Datasets/x-vs-y-for-PCA.csv'))
        c=get_cov_data(a)
        v=get_covar_eigenvectors(c)['v']
        self.assertAlmostTrue(get_PCS_vectors(a['x'],a['y'],v),np.array([
       [-0.17352675, -0.36609671,  0.21190289, -0.13750102, -0.05403107,
        -0.32075989, -0.18847975,  0.06219003,  0.07678965,  0.03967952,
         0.02226354, -0.0518399 , -0.02460134,  0.05294371, -0.04482051,
        -0.07032519, -0.01005908, -0.03319493,  0.06233973, -0.21152834,
         0.12598121,  0.24449811,  0.16592137,  0.22088582, -0.08123242,
         0.27905893,  0.16305057, -0.06983389,  0.2652078 ,  0.05668899,
         0.26239647, -0.16980096, -0.16262207, -0.1495326 , -0.20022302,
         0.00739031, -0.1682361 ,  0.01513213,  0.02431432,  0.32961041],
       [ 0.64848272,  1.6584413 , -0.98872615,  1.94077883,  1.278493  ,
        -0.39565625, -0.29259007,  1.6809071 ,  1.89875573, -0.29095872,
        -0.50378005, -0.41492376,  0.9353314 ,  0.15095545,  0.37029525,
         1.09451537,  0.36669092,  1.02210575,  1.16901004, -0.3033541 ,
         0.03865836,  1.70035248, -1.68515143, -2.19674074, -1.10511952,
         1.08109937, -2.72257491, -0.56453289, -0.3781319 , -2.73703573,
         0.52093508, -2.82387426, -0.84310462, -1.79405235, -0.14123327,
         0.52759594,  0.84342656,  2.26262539, -2.75791712,  1.75000182]]))

    def test_1_project(self):
        a=standardize_data(load_data_from_csv('Datasets/x-vs-y-for-PCA.csv'))
        c=get_cov_data(a)
        v=get_covar_eigenvectors(c)['v']
        p=get_1_projection(a,v[0])
        self.assertAlmostTrue(p,(np.array(
        [ 0.64848272,  1.6584413 , -0.98872615,  1.94077883,  1.278493  ,
       -0.39565625, -0.29259007,  1.6809071 ,  1.89875573, -0.29095872,
       -0.50378005, -0.41492376,  0.9353314 ,  0.15095545,  0.37029525,
        1.09451537,  0.36669092,  1.02210575,  1.16901004, -0.3033541 ,
        0.03865836,  1.70035248, -1.68515143, -2.19674074, -1.10511952,
        1.08109937, -2.72257491, -0.56453289, -0.3781319 , -2.73703573,
        0.52093508, -2.82387426, -0.84310462, -1.79405235, -0.14123327,
        0.52759594,  0.84342656,  2.26262539, -2.75791712,  1.75000182])))
