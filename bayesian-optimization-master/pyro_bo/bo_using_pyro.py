# pyro bo

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import pyro.distributions as dist

import ipdb
import numpy as np
import time
import numpy as np

import pyro
import pyro.contrib.gp as gp

from pyDOE import lhs
from scipy.spatial import cKDTree
#pyro.enable_validation(True)  # can help with debugging
#pyro.set_rng_seed(1)

import sys
sys.path.append("../qmc_python/")

from qmc_py import sobol_sequence

#def f(x):
#    return (6 * x - 2)**2 * torch.sin(12 * x - 4)

def f_target(x):
    """
    the target target function that we are optimizing
    """
    return torch.norm((x -0.5), dim=1)



def update_posterior(x_new, f_target, gpmodel):
    y = f_target(x_new) # evaluate f at new point.
    X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation
    y = torch.cat([gpmodel.y, y])
    gpmodel.set_data(X, y)
    #gpmodel.optimize()  # optimize the GP hyperparameters using default settings
    try:
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
        gp.util.train(gpmodel, optimizer)
    except:
        try:
            #ipdb.set_trace()
            X_new, ind_to_remove = remove_close_points(X.detach().numpy())
            y_new = np.delete(y, ind_to_remove)
            #gpmodel.set_data(X_new, y_new)
            gpmodel.set_data(torch.tensor(X_new, dtype=torch.float), y_new)
            optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
            gp.util.train(gpmodel, optimizer)
        except:
            X_new, ind_to_remove = remove_close_points(X.detach().numpy(), 0.05)
            y_new = np.delete(y, ind_to_remove)
            #gpmodel.set_data(X_new, y_new)
            gpmodel.set_data(torch.tensor(X_new, dtype=torch.float), y_new)
            optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
            gp.util.train(gpmodel, optimizer)

def lower_confidence_bound(x, kappa=2):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    
    sigma = variance.sqrt()
    return mu - kappa * sigma

def expected_improvement(x, 
    gpmodel,
    sampling_type='MC', 
    sample_size=20):
    #import ipdb; ipdb.set_trace()
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    f_star = (gpmodel.y).min()
    if sampling_type == 'MC':
        z_sample = torch.normal(torch.zeros(mu.shape[0], sample_size))
    elif sampling_type == 'RQMC':
        z_sample = sobol_sequence(sample_size, mu.shape[0], iSEED=np.random.randint(10**5), TRANSFORM=1).transpose()
        z_sample = torch.tensor(z_sample, dtype=torch.float32, requires_grad=False)
        #ipdb.set_trace()
    sigma = variance.sqrt()
    f_sample = mu.unsqueeze(1)+sigma.unsqueeze(1)*z_sample
    #import ipdb; ipdb.set_trace()
    return -torch.clamp(f_star-f_sample,0).mean(1).unsqueeze(0)

def q_expected_improvement(x_q, 
    gpmodel,
    sampling_type='MC', 
    sample_size=20):
    #import ipdb; ipdb.set_trace()
    mu_q, variance_q = gpmodel(x_q, full_cov=True, noiseless=False)
    f_star = (gpmodel.y).min()
    if sampling_type == 'MC':
        z_sample = torch.normal(torch.zeros(mu_q.shape[0], sample_size))
    elif sampling_type == 'RQMC':
        z_sample = sobol_sequence(sample_size, mu_q.shape[0], iSEED=np.random.randint(10**5), TRANSFORM=1).transpose()
        z_sample = torch.tensor(z_sample, dtype=torch.float32, requires_grad=False)
        #ipdb.set_trace()
    
    try:
        sigma = torch.cholesky(variance_q)
    except:
        small_epsilon = 0.1
        identity_mat = torch.eye(variance_q.shape[0])
        try:
            #U, S, V = torch.svd(variance_q+small_epsilon*identity_mat)
            #sigma = torch.mm(U, torch.diag(S**0.5))
            sigma = torch.cholesky(variance_q+small_epsilon*identity_mat)
        except:
            import ipdb; ipdb.set_trace()
            # in case it is still not working
    f_sample = mu_q.unsqueeze(1)+torch.mm(sigma, z_sample)
    #import ipdb; ipdb.set_trace()
    #return -torch.clamp(f_star-f_sample.min(0),0).mean(1).unsqueeze(0)
    return -torch.clamp(f_star-f_sample.min(0)[0],0).mean().unsqueeze(0)



def find_a_candidate(x_init, 
    gpmodel,
    lower_bound=0, 
    upper_bound=1, 
    sampling_type="MC", 
    sample_size=20):
    # transform x to an unconstrained domain
    #ipdb.set_trace()
    constraint = constraints.interval(lower_bound, upper_bound)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = torch.tensor(unconstrained_x_init, requires_grad=True)
    #minimizer = optim.LBFGS([unconstrained_x])
    minimizer = optim.Adam([unconstrained_x], lr=0.001)

    def closure():
        #ipdb.set_trace()
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = q_expected_improvement(x, gpmodel, sampling_type=sampling_type, sample_size=sample_size)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach()

def remove_close_points(x_init, dist=.01):
    """
    function that removes point that are too close and might cause numerical 
    problems
    input :
        x = a numpy matrix
    output : 
        the same matrix with the points removed
    """
    tree = cKDTree(x_init)
    #import ipdb; ipdb.set_trace()
    to_remove = tree.query_pairs(r=dist)
    to_remove_list = []
    if len(to_remove)>0:
        for i_set in to_remove:
            to_remove_list.append(i_set[0])
        #import ipdb; ipdb.set_trace()
        x_init = np.delete(x_init, to_remove_list, 0)
    return(x_init, to_remove_list)



def next_x(gpmodel, 
    lower_bound=0, 
    upper_bound=1, 
    num_candidates=20, 
    sampling_type="RQMC", 
    sample_size=20,
    q_size=20):
    candidates = []
    values = []
    #import ipdb; ipdb.set_trace()
    dim = gpmodel.X.shape[1]
    # TODO: find a way to get a latin hypercube desing with respect to number of points, 
    # dimension of X and q points
    #x_init_points = torch.tensor(lhs(n=dim, samples=num_candidates, criterion='maximin'), dtype=torch.float)
    
    #x_init = x_init_points[:1,:]#gpmodel.X[-1:,:]
    for i in range(num_candidates):
        x_init = lhs(n=dim, samples=q_size, criterion='maximin')
        # remove points that are too close to each other
        x_init, __ = remove_close_points(x_init)


        x_init_points = torch.tensor(x_init, dtype=torch.float)

        #import ipdb; ipdb.set_trace()
        x_init = x_init_points #x_init = x_init_points[i,:].unsqueeze(0)
        x = find_a_candidate(x_init, gpmodel, lower_bound, upper_bound, sampling_type=sampling_type, sample_size=sample_size)
        y = q_expected_improvement(x, gpmodel, sampling_type=sampling_type, sample_size=sample_size)
        
        candidates.append(x)
        values.append(y)
        #import ipdb; ipdb.set_trace()
        #x_init = x.new_empty(gpmodel.X.shape[1]).uniform_(lower_bound, upper_bound).unsqueeze(0)
    argmin = torch.min(torch.cat(values), dim=0)[1].item()
    #import ipdb; ipdb.set_trace()
    return candidates[argmin]

def plot(gs, xmin, xlabel=None, with_title=True):
    xlabel = "xmin" if xlabel is None else "x{}".format(xlabel)
    Xnew = torch.linspace(-0.1, 1.1)
    ax1 = plt.subplot(gs[0])
    ax1.plot(gpmodel.X.numpy(), gpmodel.y.numpy(), "kx")  # plot all observed data
    with torch.no_grad():
        loc, var = gpmodel(Xnew, full_cov=False, noiseless=False)
        sd = var.sqrt()
        ax1.plot(Xnew.numpy(), loc.numpy(), "r", lw=2)  # plot predictive mean
        ax1.fill_between(Xnew.numpy(), loc.numpy() - 2*sd.numpy(), loc.numpy() + 2*sd.numpy(),
                         color="C0", alpha=0.3)  # plot uncertainty intervals
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_title("Find {}".format(xlabel))
    if with_title:
        ax1.set_ylabel("Gaussian Process Regression")

    ax2 = plt.subplot(gs[1])
    with torch.no_grad():
        # plot the acquisition function
        #import ipdb; ipdb.set_trace()
        ax2.plot(Xnew.numpy(), expected_improvement(Xnew, 1000).numpy().flatten())
        # plot the new candidate point
        ax2.plot(xmin.numpy(), expected_improvement(xmin, 1000).numpy().flatten(), "^", markersize=10,
                 label="{} = {:.5f}".format(xlabel, xmin.item()))
    ax2.set_xlim(-0.1, 1.1)
    if with_title:
        ax2.set_ylabel("Acquisition Function")
    ax2.legend(loc=1)

def random_search(params_data, outer_loop_steps=10, q_size=5):
    # compare to benchmark of random sampling
    n_total_samples = outer_loop_steps*q_size
    f_target = params_data['f_target']
    X = params_data['X']
    y = params_data['y']
    dim = params_data['dim']
    
    X_new = lhs(n=dim, samples=n_total_samples, criterion='maximin')
    y_new = f_target(X_new)
    y_all = np.concatenate([y,y_new])
    X_all = np.concatenate([X,X_new])
    y_min = np.min(y_all)
    X_min = X_all[np.argmin(y_all),:]
    res_dict = {'X_exp' : X_all, 'y_exp' :y_all,
                'X_min': X_min, 'y_min' : y_min
                }
    print('random search: minumum found at x=%s, and y=%s' %(X_min, y_min))
    return(None, res_dict)

def run_bo_pyro(params_bo, params_data, outer_loop_steps=10, q_size=5):
    """
    run the bo 
    """
    X = params_data['X']
    y = params_data['y']
    #import ipdb; ipdb.set_trace()
    gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim=params_data['dim']),
                                 noise=torch.tensor(0.1), jitter=10**(-1))
    gpmodel.kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    gpmodel.kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))

    sampling_type = params_bo['sampling_type']
    sample_size = params_bo['sample_size']
    f_target = params_data['f_target']
    print('run model with %s and %s samples' % (sampling_type, sample_size))
    start_time = time.time()
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)

    #gpmodel.optimize()
    y_list_min  = []
    x_list_min  = []
    for i in range(outer_loop_steps):
        print('approach %s, sample size %s, outer loop step %s of %s'% (sampling_type, sample_size, i, outer_loop_steps))
        xmin = next_x(gpmodel, sampling_type=sampling_type, sample_size=sample_size, q_size=q_size)
        print("next points evaluated:")
        #import ipdb; ipdb.set_trace()
        print(xmin, f_target(xmin))
        # update the posterior with the new point found
        #TODO: do we evaluate the target function at all points??
        update_posterior(xmin, f_target, gpmodel)
        # add to the list of best sampled points
        val_y, ind = gpmodel.y.min(0)
        val_x = gpmodel.X[ind,:]
        y_list_min.append(val_y.detach().numpy())
        x_list_min.append(val_x.detach().numpy())
        print("best points so far:")
        print(x_list_min[-1], y_list_min[-1])
    


    print("run time %s seconds" %(time.time() -start_time))
    res_dict = {'X_exp' : gpmodel.X.detach().numpy(), 'y_exp' :gpmodel.y.numpy(),
                'X_min': np.array(x_list_min), 'y_min' : np.array(y_list_min)
                }
    return gpmodel, res_dict

if __name__ == '__main__':
    dim = 3
    np.random.seed(42)
    X = np.random.random(size=(7, dim))
    X = torch.tensor(X, dtype=torch.float)
    y = f_target(X)

    params_data = {
        'X' : X,
        'y' : y,
        'dim' : dim,
        'noise' : 0.01, 
        'f_target' : f_target
        }

    params_bo_mc = {
        'sampling_type' : 'MC', 
        'sample_size' : 20,
        'num_candidates' : 20
    }

    params_bo_rqmc = {
        'sampling_type' : 'RQMC', 
        'sample_size' : 20,
        'num_candidates' : 20
    }

    res_model_mc, mc_dict = run_bo_pyro(params_bo_mc, params_data)
    res_model_rqmc, rqmc_dict = run_bo_pyro(params_bo_rqmc, params_data)

    import ipdb; ipdb.set_trace()

if False: 
    do_plot = False
    if do_plot: 
        plt.figure(figsize=(12, 30))
        outer_gs = gridspec.GridSpec(5, 2)
    gpmodel.optimize()
    for i in range(15):
        print(i)
        xmin = next_x()
        if do_plot:
            gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i])
            plot(gs, xmin, xlabel=i+1, with_title=(i % 2 == 0))
        update_posterior(xmin)
        #xmin = next_x()
        #update_posterior(xmin)
    if do_plot: 
        plt.show()
        plt.plot(gpmodel.X.numpy(), gpmodel.y.numpy())
    import ipdb; ipdb.set_trace()