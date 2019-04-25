# RL-BrachyEnv
RL Prj: Treatment Planning in Brachytherapy via Reinforcement Learning

Code for Discrete & Continuous scenario is in Tabular & PolicyGradient folder respectively.

do python training.py to start the training in each case.

The hyperparameters are listed at the begining of each training.py file, for continuous case, the important ones are

constraint: in [0, 1, 2] corresponding to c0, c1 & c2

std_scale: initial logstd scale for value and policy network

using_pcnt: boolean indicating whether includes pseudo counts


for iterative evaluate density function for pseudo counts

var: initial variance

scale: beta
