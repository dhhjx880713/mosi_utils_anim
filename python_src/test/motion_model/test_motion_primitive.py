# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:42:34 2015

@author: erhe01

All Unittests for the MotionPrimitive class
"""


import os
import sys
ROOTDIR = os.sep.join(['..'] * 2)
sys.path.append(ROOTDIR + os.sep)
TESTLIBPATH = ROOTDIR + os.sep + 'test'
sys.path.append(TESTLIBPATH)
sys.path.append(ROOTDIR + os.sep + 'morphablegraphs/construction')
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/motion_primitive'
from libtest import params, pytest_generate_tests
import numpy as np
from morphablegraphs.motion_model.motion_primitive import MotionPrimitive

class TestMotionPrimitive(object):
    """MotionPrimitive test class"""

    def setup_method(self, method):
        testfile = TEST_DATA_PATH + os.sep + 'walk_leftStance_quaternion_mm.json'
        self.mp =  MotionPrimitive(testfile)
        self.number_of_samples = 1000

    def test_init(self):
        """ Test if MotionPrimitive has been correctly initialized """

        assert "eigen_vectors" in self.mp.s_pca.keys() and "eigen_vectors" \
                                                        in self.mp.t_pca.keys()
                                                        
        assert len(self.mp.gmm.means_[0]) == len(self.mp.s_pca["eigen_vectors"])\
                                            +len(self.mp.t_pca["eigen_vectors"].T)

    def test_inverse_spatial_pca_shape(self):
        """ Test if the inverse spatial pca produces an array with the expected
            shape from a random sample
        """
        sample = np.ravel(self.mp.gmm.sample())
        alpha = sample[:len(self.mp.s_pca["eigen_vectors"])]
        coefs = self.mp._inverse_spatial_pca(alpha)
        assert not np.isnan(coefs).any()
        assert coefs.shape == (self.mp.s_pca["n_basis"], self.mp.s_pca["n_dim"])
        
        
        
    param_inverse_spatial_pca = [{"alpha": [3.5240015317642643, 0.40938595946091794, -0.35035145952216606, 0.53743024695364028,
                                            -0.14766110820311193, 0.18499043544137223, 0.099928929974349545, 0.35874412680357559,
                                            -0.38055234638285262],
                                  "res": [[[  9.28269086e-02,   9.98669021e+01,  -2.60228354e-01,
         -3.33139193e-02,   6.90238384e-01,   7.16269005e-01,
          3.55939486e-02,   9.99554050e-01,  -2.34527307e-02,
         -1.23067706e-02,   1.19580034e-02,   1.00000000e+00,
          1.57437462e-09,  -7.21663693e-10,  -4.00720598e-10,
          9.75006060e-01,   2.36704160e-02,  -2.02373718e-01,
          5.83489409e-02,   9.90615300e-01,   3.53109549e-02,
          9.12598158e-02,  -3.43584128e-02,  -8.41586163e-02,
         -4.68580663e-01,   8.68688836e-01,   3.43454137e-03,
          9.14320551e-01,   3.01152218e-01,   2.20624254e-01,
          1.82381983e-01,   9.71073669e-01,   9.31880132e-07,
          2.34577797e-01,   4.04583655e-07,   6.97009054e-01,
         -7.15348497e-01,  -3.44566598e-02,   7.56982199e-02,
         -7.55941920e-03,   5.17318169e-01,   8.55874828e-01,
         -9.19670934e-02,   9.48653398e-01,  -3.47449451e-01,
          1.57432390e-01,  -2.12731089e-01,   9.81878750e-01,
         -2.56322059e-07,   1.90287271e-01,  -5.14112427e-08,
          7.40413975e-01,   6.51782583e-01,  -1.65753451e-01,
         -7.60027420e-02,  -1.17659120e-01,  -3.13936706e-02,
         -1.08347128e-01,   9.87625808e-01,   9.17499900e-01,
         -2.08354677e-11,   4.19059882e-01,  -6.73306414e-09,
          9.95500073e-01,   9.91190060e-02,  -8.20382620e-03,
          3.87727871e-02,   1.87709395e-02,   8.40967887e-02,
          8.46752511e-02,   9.93878968e-01,   9.91172050e-01,
         -2.08037172e-09,   1.28776929e-01,  -4.24026123e-09,
          9.99217597e-01,  -1.37464169e-02,   3.61898622e-03,
          3.25572594e-02]], [[  1.86359147e+00,   1.00980212e+02,  -4.42840269e+00,
         -5.83570738e-02,   6.87022093e-01,   7.22227005e-01,
          2.12379970e-02,   9.99621998e-01,  -1.91246594e-02,
         -2.03380641e-02,   1.59023096e-02,   1.00000000e+00,
         -1.50828730e-09,  -9.24436612e-10,   3.66996892e-10,
          9.74704342e-01,   2.49068780e-02,  -2.02481042e-01,
          6.84344079e-02,   9.90771631e-01,   3.06272451e-02,
          9.44914553e-02,  -3.04632495e-02,  -7.76975380e-02,
         -4.66731346e-01,   8.70671370e-01,   4.83455108e-03,
          9.12651458e-01,   3.08590042e-01,   2.06619155e-01,
          1.95099570e-01,   9.77431768e-01,   7.50806289e-07,
          2.11102536e-01,   2.89764223e-07,   6.93535459e-01,
         -7.13333283e-01,  -4.44254167e-02,   8.36608537e-02,
         -1.60772957e-02,   5.21406723e-01,   8.53473383e-01,
         -8.62629045e-02,   9.51740336e-01,  -3.51085227e-01,
          1.72984394e-01,  -1.91075478e-01,   9.78425276e-01,
         -3.35981798e-07,   2.01387141e-01,  -5.96694010e-08,
          7.37763536e-01,   6.56227316e-01,  -1.53626544e-01,
         -7.08510955e-02,  -1.35610499e-01,   1.62156408e-02,
         -1.14193023e-01,   9.85797487e-01,   8.81886350e-01,
         -1.76314977e-09,   4.67169851e-01,  -7.80268059e-09,
          9.94547963e-01,   6.04344064e-02,   1.00786842e-02,
          5.06176456e-02,   3.60110605e-03,   3.75082742e-02,
          9.43656586e-02,   9.97717730e-01,   9.91252005e-01,
          2.28246177e-09,   1.26179928e-01,  -1.71197754e-09,
          9.99639551e-01,  -1.26712612e-03,  -2.10936355e-02,
          2.28432199e-02]],[[  3.26918578e+00,   1.01878661e+02,  -1.15897705e+01,
         -6.65513432e-02,   6.96001862e-01,   7.14651383e-01,
          2.94937478e-02,   9.99610431e-01,   4.07478134e-03,
         -2.79273337e-02,   1.39871729e-02,   1.00000000e+00,
          5.55158300e-10,   3.75324413e-12,  -2.90842197e-10,
          9.72571836e-01,   4.07707496e-02,  -2.14930222e-01,
          5.34169014e-02,   9.91429231e-01,   2.64104466e-02,
          9.01905858e-02,  -2.00988243e-02,  -7.04453293e-02,
         -4.65359524e-01,   8.72173601e-01,   1.77357862e-02,
          9.20586390e-01,   2.99703183e-01,   1.70293739e-01,
          2.08531372e-01,   9.82828699e-01,   7.43339393e-08,
          1.90872550e-01,   4.40695261e-08,   7.01969563e-01,
         -7.07282411e-01,  -6.36626350e-02,   9.84958518e-02,
         -3.67977065e-02,   5.22019349e-01,   8.52957002e-01,
         -8.02030965e-02,   9.56200333e-01,  -3.46809045e-01,
          2.11252520e-01,  -1.52552256e-01,   9.77549852e-01,
         -9.06646699e-07,   2.07617405e-01,  -2.95303381e-07,
          7.32573972e-01,   6.66163784e-01,  -1.34247739e-01,
         -5.02634168e-02,  -1.12555534e-01,   9.78162283e-02,
         -1.35692549e-01,   9.80668188e-01,   8.59305206e-01,
          2.25367816e-10,   5.28886557e-01,  -4.38196274e-09,
          9.97610997e-01,   2.17567090e-03,  -5.78934245e-02,
          4.32201861e-02,   9.11420313e-03,  -2.93529989e-02,
          1.04270338e-01,   9.97869510e-01,   9.95916231e-01,
         -6.39959934e-09,   1.12970901e-01,  -2.94513061e-09,
          9.99234455e-01,  -1.51809739e-02,  -5.67213457e-02,
          2.49623580e-02]],[[  3.85144978e+00,   1.01112075e+02,  -2.14043745e+01,
         -6.51948744e-02,   7.07830513e-01,   7.12979883e-01,
          6.72647998e-03,   9.98990482e-01,   2.29130744e-02,
         -2.54586053e-02,   2.41622395e-02,   1.00000000e+00,
         -4.77209364e-09,  -1.17457620e-09,   4.35989991e-10,
          9.72276207e-01,   5.53119786e-02,  -2.16263106e-01,
          2.41017998e-02,   9.92002073e-01,   2.57922864e-02,
          7.95963280e-02,  -3.34109294e-02,  -5.83364274e-02,
         -4.66475584e-01,   8.72312467e-01,   2.48775553e-02,
          9.23533819e-01,   2.91375790e-01,   1.36027546e-01,
          2.27072070e-01,   9.82896769e-01,  -7.83381490e-08,
          1.92876312e-01,   2.63804686e-10,   6.91962529e-01,
         -7.16556219e-01,  -5.28438854e-02,   9.36383819e-02,
         -5.85435419e-02,   5.25382505e-01,   8.49214353e-01,
         -7.59934310e-02,   9.69210721e-01,  -3.10722615e-01,
          2.41168463e-01,  -1.01937681e-01,   9.66105363e-01,
         -1.43979171e-06,   2.54931963e-01,  -5.06677665e-07,
          7.22008327e-01,   6.91052398e-01,  -7.49172831e-02,
         -2.62979303e-03,  -1.96664799e-02,   1.56195265e-01,
         -1.72282821e-01,   9.73293152e-01,   1.00010308e+00,
         -2.95408051e-09,   2.21546188e-01,  -2.17896659e-10,
          9.97812006e-01,   7.51726725e-04,  -5.18392719e-02,
         -3.21604776e-04,   6.16308946e-03,  -8.85315841e-02,
          1.26953420e-01,   9.88130593e-01,   9.95752274e-01,
          4.64855371e-10,   8.93047498e-02,  -1.43788665e-11,
          9.95014989e-01,  -1.48365334e-02,  -1.09587615e-01,
          1.34065195e-02]],[[  3.47535910e+00,   9.79879113e+01,  -3.87881727e+01,
         -3.71255880e-02,   7.00889442e-01,   7.23305168e-01,
          4.53536833e-03,   9.99281834e-01,   3.21916779e-02,
          5.29556672e-04,   1.80322403e-02,   1.00000000e+00,
          2.90864324e-09,  -1.17554948e-10,   1.52353056e-10,
          9.73385942e-01,   6.73262319e-02,  -2.08046801e-01,
          2.89970680e-03,   9.90456311e-01,   3.30180273e-02,
          8.99058223e-02,  -3.47338680e-02,  -5.14553300e-02,
         -4.69454855e-01,   8.70671595e-01,   2.47366851e-02,
          9.22729018e-01,   2.86218305e-01,   1.29847832e-01,
          2.36413949e-01,   9.84389186e-01,   2.14503602e-07,
          1.84364556e-01,   1.06593722e-07,   6.93889968e-01,
         -7.13522700e-01,  -6.38627090e-02,   1.00913329e-01,
         -4.52556312e-02,   5.19523387e-01,   8.54360292e-01,
         -7.45693129e-02,   9.57483574e-01,  -3.19754663e-01,
          2.50333279e-01,  -1.23154657e-01,   9.58900849e-01,
         -9.90651639e-07,   2.81299580e-01,  -4.52623503e-07,
          7.26647366e-01,   6.85909337e-01,  -4.19252020e-02,
          2.20249872e-02,   8.20374551e-03,   1.48597938e-01,
         -1.30488169e-01,   9.78108534e-01,   9.97269334e-01,
          1.17661750e-09,   2.75585183e-02,   6.48160692e-09,
          9.98228096e-01,  -7.44529457e-02,   1.99762610e-03,
          1.29205050e-02,   2.00975048e-02,  -1.29974548e-01,
          1.34422659e-01,   9.81374441e-01,   9.88809514e-01,
         -8.00725074e-11,   1.58544503e-01,  -6.26650952e-09,
          9.79576261e-01,  -5.14751786e-02,  -1.91529833e-01,
          4.49502724e-02]],[[  6.82313803e-01,   9.90575204e+01,  -4.93181022e+01,
         -3.72647076e-02,   7.13695672e-01,   7.10932327e-01,
          2.49580725e-02,   9.99257408e-01,   3.46456887e-02,
          9.00725631e-04,   1.01864598e-03,   1.00000000e+00,
         -3.01382447e-09,  -1.62529842e-09,  -4.27264886e-10,
          9.73650203e-01,   6.81147399e-02,  -2.02986434e-01,
         -2.37037756e-02,   9.90099686e-01,   3.63173027e-02,
          9.06841044e-02,  -2.70956065e-02,  -5.26273982e-02,
         -4.73701418e-01,   8.67362447e-01,   2.54714611e-02,
          9.25414091e-01,   2.86947643e-01,   1.32300406e-01,
          2.28237084e-01,   9.84549976e-01,   1.00466635e-07,
          1.81899372e-01,   5.66722041e-08,   6.95870645e-01,
         -7.13304310e-01,  -6.55561628e-02,   1.00404296e-01,
         -2.95425378e-02,   5.10707530e-01,   8.60722791e-01,
         -7.24855333e-02,   9.43544764e-01,  -3.40379195e-01,
          2.35028960e-01,  -1.71030945e-01,   9.69352470e-01,
         -1.70532434e-06,   2.48157310e-01,  -7.17319966e-07,
          7.38900792e-01,   6.71588095e-01,  -9.22050032e-02,
         -1.71835997e-02,   1.17657886e-02,   1.09511682e-01,
         -1.10592523e-01,   9.87443811e-01,   9.93126313e-01,
         -6.39992803e-10,   1.34030570e-01,  -2.43674298e-09,
          9.99248948e-01,  -8.61094395e-02,   3.60067151e-02,
          4.97493529e-02,   8.82906731e-02,  -8.93600494e-02,
          1.40155550e-01,   9.82673813e-01,   9.61831562e-01,
          9.33237831e-10,   2.93796272e-01,  -9.46149163e-09,
          9.98152322e-01,  -8.74293184e-02,  -6.02376156e-02,
          4.14326996e-02]],[[ -5.82143871e-01,   9.73514449e+01,  -5.42811289e+01,
         -4.43486348e-02,   7.16101764e-01,   7.09148405e-01,
          2.39229540e-02,   9.99477135e-01,   2.90526153e-02,
         -3.04662730e-03,  -1.70154726e-03,   1.00000000e+00,
          1.24349061e-09,  -1.88525877e-10,   8.20662620e-11,
          9.73730810e-01,   6.42474550e-02,  -2.01632819e-01,
         -3.23430371e-02,   9.89922939e-01,   3.94680289e-02,
          9.09078379e-02,  -3.02010705e-02,  -5.94559867e-02,
         -4.76893068e-01,   8.64602215e-01,   2.23354190e-02,
          9.26061623e-01,   2.90892802e-01,   1.44794158e-01,
          2.16121559e-01,   9.82637330e-01,   2.17597810e-07,
          1.88152239e-01,   1.02283948e-07,   6.93223155e-01,
         -7.17004714e-01,  -6.37005751e-02,   9.85954946e-02,
         -2.47739210e-02,   5.08096114e-01,   8.62007224e-01,
         -7.44290730e-02,   9.38681494e-01,  -3.48298321e-01,
          2.25782266e-01,  -1.86913818e-01,   9.72833998e-01,
         -1.32238549e-06,   2.34187808e-01,  -4.82978749e-07,
          7.44874935e-01,   6.61643394e-01,  -1.16794991e-01,
         -3.70013035e-02,   2.54028887e-02,   7.33459757e-02,
         -1.12301977e-01,   9.90389588e-01,   9.93929912e-01,
          1.25446208e-10,   1.01121794e-01,   1.43937742e-09,
          9.97941002e-01,  -8.85991299e-02,  -8.97381723e-04,
          4.05488369e-02,   1.09133620e-01,  -4.54884283e-02,
          1.40400582e-01,   9.83034381e-01,   9.30230795e-01,
         -3.91028278e-09,   3.77973015e-01,  -1.08728492e-08,
          9.99985723e-01,  -7.48755005e-02,  -1.44334131e-02,
          3.11597804e-02]]]}]


    @params(param_inverse_spatial_pca)
    def test_inverse_spatial_pca_result(self,alpha, res):
        """ Test if the inverse spatial pca produces the expected result
        """
        coefs = self.mp._inverse_spatial_pca(np.array(alpha)).tolist()
        coefs = np.ravel(coefs)
        res = np.ravel(res)
        assert len(coefs) == len(res)
        for i in xrange(len(coefs)):
            assert  round(coefs[i], 5) == round(res[i], 5)
      
        
    def test_inverse_time_pca(self):
        """ Test if the inverse temporal pca produces a strictly monotonously
            increasing vector using multiple samples
        """
        for s in xrange(self.number_of_samples):
            sample = np.ravel(self.mp.gmm.sample())
            gamma = sample[len(self.mp.s_pca["eigen_vectors"]):]
            t = self.mp._inverse_temporal_pca(gamma)
            if not np.all([t[i]>t[i-1] for i in xrange(len(t)) if i > 0]):
                print s,t
            assert not np.isnan(t).any()
            assert np.all([t[i]>t[i-1] for i in xrange(len(t)) if i > 0])
            assert t[-1] == self.mp.n_canonical_frames -1
            
            
        

