
Download and run
1. Download zip or git clone.
2. 在linux下运行，推荐使用GPU。
3. pip install -r requirements.txt
4. 如果GPU版本的jax或jaxlib在安装中出现问题，与本代码无关，可在网络上寻找解决方案。If the GPU version of 'jax' or 'jaxlib' is having problems with the installation, it is not related to this code, and you can look for a solution on the network.
5. This program is temporarily run using a script: 'python example/input.py', 
modifying the parameters in the 'input.py' file,


Add loss function:
1. 'opt_coil/lossfunction.py' : Write function and add it to 'loss_value' and 'loss_save'.
2. If the required parameters are not given, they should be calculated in 'opt_coil/coilset.py' and added to the return of 'cal_coil', also change the first line of 'loss_value' and 'loss_save'.
3. 'example/input.py' : Add a weight ('weight_loss') or target ('target_loss') value.


Some details of the Settings are not given in the input file, if you need to adjust the following features, please change in .py file

-**f_B:** 'opt_coil/lossfunction.py', you can choose 'f_B~|bn|' or 'f_B~(bn)^2' in 'line129' or 'line131'
-**B_reg:** 'HTS/B_self.py',  if you want to use the accurate formula, add 'line 140-144'.
-**strain** 'HTS/hts_strain.py', 'dtheta' is calculated from 'arcsin(theta)' or 'arccos(theta)', their performance in optimization is somewhat different, you can change 'line 28-33' or 'line 35-40'.







