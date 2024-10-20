
写完之后再确认一下位置




Add loss function:




Some details of the Settings are not given in the input file, if you need to adjust the following features, please change in .py file

-**f_B:** 'opt_coil/lossfunction.py', you can choose 'f_B~|bn|' or 'f_B~(bn)^2' in 'line129' or 'line131'
-**B_reg:** 'HTS/B_self.py',  if you want to use the accurate formula, add 'line 140-144'.
-**strain** 'HTS/hts_strain.py', 'dtheta' is calculated from 'arcsin(theta)' or 'arccos(theta)', their performance in optimization is somewhat different, you can change 'line 28-33' or 'line 35-40'.























