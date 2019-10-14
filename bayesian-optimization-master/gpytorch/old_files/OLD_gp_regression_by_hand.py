# gp regression, coded by hand
from __future__ import print_function
from __future__ import division


import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from functions_bo import *

x_range = 4
sigma_prior = 0.1
sample_size = 20
dim_x = 4
starting_points = 10
mkernel = MaternKernel(ard_num_dims=dim_x) # this arguments tell the kernel that our input is 2d

def sample_points(n, dim, x_range, requires_grad=False):
    x = 2*x_range*torch.rand(n, dim, requires_grad=requires_grad)-x_range
    return(x)


def target_function(x):
    return(-torch.norm(x, dim=1)**2)


x_train = sample_points(starting_points, dim_x, x_range)
y_train = target_function(x_train)


parameters_model = {'sigma_prior': sigma_prior, 'sample_size' : sample_size }
parameters_data = {
    'dim_x' : dim_x, 
    'y_train' : y_train,
    'x_train' : x_train,
    'x_to_evaluate' : torch.ones(1, dim_x, 
    'x_range' : x_range, 
    'sample_points': sample_points)
}
class_ei = OneExpectedImprovement(parameters_model, parameters_data)

class_ei.fit_gp()



#par, value = maximize_one_ei(class_ei, training_iter = 1000)
#y_new_eval = target_function(par)
#class_ei.update_gp(y_new_eval, par)
#starting_point = torch.nn.Parameter(x_range*torch.randn(1, dim_x, requires_grad=True))
#class_ei.my_param = starting_point




plt.scatter(class_ei.x_train.data.numpy()[:,0], class_ei.y_train.data.numpy(), label='starting point')
plt.show()


bayesian_optimization(class_ei, 60)
import ipdb; ipdb.set_trace()

plt.scatter(class_ei.x_train.data.numpy()[:,0], class_ei.y_train.data.numpy(), label='starting point')
plt.show()
#out_covar = covar_pred(X_to_evaluate, y_train, x_train)
#out_mean = mean_pred(X_to_evaluate, y_train, x_train)
if False: 
    for t in range(1000):
        import ipdb; ipdb.set_trace()
        out_mean = mean_pred(X_to_evaluate, y_train, x_train)
        out_mean.backward(y_train)
        with torch.no_grad():
            X_to_evaluate += 0.001 * X_to_evaluate.grad
            X_to_evaluate.grad.zero_()
# play around and make some predictions
m_multistart = 10
import copy
import numpy as np
X_test = (torch.rand((m_multistart, dim_x), requires_grad=False)*2*x_range)-x_range
X_test_copy = copy.deepcopy(X_test)
#import ipdb; ipdb.set_trace()
#X_test = torch.tensor([9.], requires_grad=True, dtype=torch.float)
multistart_loss = []
multistart_x_test = []
all_points_list = []
all_loss_list = []
for i_multistart in range(m_multistart):
    X_test_inter = X_test[i_multistart,:].unsqueeze(0)
    X_test_inter.requires_grad = True
    #import ipdb; ipdb.set_trace()
    for t in range(1000):
        import ipdb; ipdb.set_trace()
        one_expected_improvement(X_test, y_train, x_train, 1000)
        loss = one_expected_improvement(X_test_inter, y_train, x_train, 1000)
        all_loss_list.append(loss.data.numpy())
        all_points_list.append(copy.deepcopy(X_test_inter).data.numpy())
        #loss = mean_pred(X_test_inter, y_train, x_train)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            X_test_inter += 0.01 * X_test_inter.grad
            #print(X_test_inter.grad)
            #store_grad = X_test_inter.grad.data.numpy()
            X_test_inter.grad.zero_()
    print(X_test_inter, loss)
    multistart_loss.append(loss.data.numpy())
    multistart_x_test.append(X_test_inter.data.numpy())
#import ipdb; ipdb.set_trace()
if True: 
    x_support = torch.linspace(-x_range, x_range, 200, requires_grad=False).unsqueeze(1)
    confidence_bound = covar_pred(x_support, y_train, x_train).diag()
    predicted_y = mean_pred(x_support, y_train, x_train).squeeze()
    #plt.scatter(x_train.data.numpy(), y_train.data.numpy(), label='true points')
    plt.scatter(np.array(all_points_list), np.array(all_loss_list), label='trajectories')
    plt.scatter(np.array(multistart_x_test), np.array(multistart_loss), label='optimized point')
    plt.scatter(np.array(X_test_copy.data.numpy()), one_expected_improvement(X_test_copy, y_train, x_train, 1000).squeeze().data.numpy(), label='starting point')
    #plt.plot(x_support.data.numpy(), predicted_y.data.numpy(), label='predicted_points')
    #plt.plot(x_support.data.numpy(), (predicted_y+2*confidence_bound).data.numpy(), label='upper bound')
    #plt.plot(x_support.data.numpy(), (predicted_y-2*confidence_bound).data.numpy(), label='lower bound')
    plt.plot(x_support.data.numpy(), one_expected_improvement(x_support, y_train, x_train, 1000).data.numpy(), label='expected improvement')
    plt.legend()
    plt.show()



import ipdb; ipdb.set_trace()