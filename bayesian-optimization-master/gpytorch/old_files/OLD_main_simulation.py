# main that runs the simulation several times

from __future__ import print_function
from __future__ import division


import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#import seaborn as sns
import numpy as np
from functions_bo import *


x_range = 4
sigma_prior = 0.1
sample_size = 10
steps_outer_loop = 50
steps_inner_loop=10
multistarts=10

M_rep = 40
dim_x = 3
starting_points = 10
mkernel = MaternKernel(ard_num_dims=dim_x) # this arguments tell the kernel that our input is 2d

def f_sample_points(n, dim, x_range, requires_grad=False):
    x = 2*x_range*torch.rand(n, dim, requires_grad=requires_grad)-x_range
    return(x)


def target_function(x):
    return(-torch.norm(x, dim=1)**2)


x_train = f_sample_points(starting_points, dim_x, x_range)
y_train = target_function(x_train)


parameters_model_rqmc = {'sigma_prior': sigma_prior, 'sample_size' : sample_size, 'sampling_type' : 'RQMC', 'kernel' : mkernel}
parameters_model_mc = {'sigma_prior': sigma_prior, 'sample_size' : sample_size, 'sampling_type' : 'MC', 'kernel' : mkernel}
parameters_data = {
    'dim_x' : dim_x, 
    'y_train' : y_train,
    'x_train' : x_train,
    'x_to_evaluate' : torch.ones(1, dim_x), 
    'x_range' : x_range, 
    'f_sample_points': f_sample_points, 
    'target_function' : target_function,
}
#class_ei = OneExpectedImprovement(parameters_model, parameters_data)
#class_ei.fit_gp()


#plt.scatter(class_ei.x_train.data.numpy()[:,0], class_ei.y_train.data.numpy(), label='starting point')
#plt.show()


#bayesian_optimization(class_ei, 60)
#import ipdb; ipdb.set_trace()

#plt.scatter(class_ei.x_train.data.numpy()[:,0], class_ei.y_train.data.numpy(), label='starting point')
#plt.show()
def repeat_sampling(sample_sizes_list, M_rep, sampling_types_list, parameters_model, parameters_data):
    dict_all_res = {str(i_sample_size) : {} for i_sample_size in sample_sizes_list} # list to fill up
    for i_sampling_type in sampling_types_list:
        for i_sample_size in sample_sizes_list: 
            list_res = []
            for m_rep in range(M_rep):
                print('mrep = %s for sampling type %s and sample size %s' % (m_rep, i_sampling_type, i_sample_size))
                parameters_model['sampling_type'] = i_sampling_type # set sampling type to MC or RQMC
                parameters_model['sample_size'] = i_sample_size
                class_ei = OneExpectedImprovement(parameters_model, parameters_data)
                class_ei.fit_gp()
                res = bayesian_optimization(class_ei, steps_outer_loop=60, steps_inner_loop=20, multistarts=10)
                list_res.append(res)
                del class_ei
            y_res = np.array([i[0].detach().numpy() for i in list_res])
            x_res = np.array([i[1].detach().numpy() for i in list_res])
            dict_all_res[str(i_sample_size)][i_sampling_type] = {'x' : x_res, 'y' : y_res}
    return dict_all_res

sample_sizes_list = [5, 10, 20]#, 100]
M_rep = 40
sampling_types_list = ['MC', 'RQMC']
dict_all_res = repeat_sampling(sample_sizes_list, M_rep, sampling_types_list, parameters_model_mc, parameters_data)
import pickle
pickle.dump(dict_all_res, open('first_sim_bo_M_%s.pkl' % M_rep, 'wb'), protocol=2)
import ipdb; ipdb.set_trace()

list_res_plotting = []
labels_list = []
for i_sampling_type in sampling_types_list:
    for i_sample_size in sample_sizes_list: 
        list_res_plotting.append(dict_all_res[str(i_sample_size)][i_sampling_type]['y'])
        labels_list.append(str(i_sample_size)+'_'+i_sampling_type)
import ipdb; ipdb.set_trace()
#res_list = [res['y'] for res in ]
plt.boxplot(list_res_plotting, labels=labels_list)
plt.savefig('boxplot_compare_M_%s.png' % M_rep)
plt.show()
