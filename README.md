These are the Python code I used in the project to estimate pulse wave velocity, a risk factor of cardiovascular disease, using radial pressure wave, a peripheral pulse wave signal. These tools can be modified and applied to estimate other risk factors for diseases or any other subject of interests using pulse wave signals. 

The article that is associated to these code is currently preprinted on medRxiv: doi: https://doi.org/10.1101/2020.11.29.20239962.

LassoCV.py and identifyOutlinerwithPCA.py are the code used for data pre-processing. 

GPR.py is the code for conducting Gaussian process regression, which different covariance functions and their combinations have been tested.

SVR_CV.py is the code used to find the hyper-parameters for support vector regression (SVR.py), which the Optunity package (https://optunity.readthedocs.io/en/latest/) has been used. 

RandomForest.py and GBosst.py are the code for random forest regression and gradient boosting regression, respectively.

RNN_LSTM.py and RNN_GRU.py are the code for recurrent nerual network with long short-term memory and gated recurrent unit, respectively. Tensorflow 2.0 is needed for these two deep learning algorithms.

bland_altman_plot.py is adapted from https://github.com/josesho/bland_altman.

All input data in the code are excluded from the depository due to GDPR.
