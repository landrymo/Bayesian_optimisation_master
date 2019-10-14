# fit the gp and test it
from __future__ import print_function
from __future__ import division


import math
import matplotlib
import time
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#import seaborn as sns
import numpy as np
from pyDOE import lhs
from scipy.spatial import cKDTree

import torch
import gpytorch

import sys
sys.path.append("../qmc_python/")

from qmc_py import sobol_sequence

# Training data is 11 points in [0,1] inclusive regularly spaced
dim = 1
#train_x = torch.randn((20, dim))
#train_x = torch.linspace(0, 1, 5).unsqueeze(1)
sorted, indices = torch.sort(torch.rand(20))
train_x = sorted.unsqueeze(1)
# True function is sin(2*pi*x) with Gaussian noise
def f_target_norm(x):
    """
    the target target function that we are optimizing
    """
    return torch.norm((x -0.5), dim=1)**4

def target_y(train_x):
    train_y = torch.sin(train_x.sum(1)**2 * (2 * math.pi)) + torch.randn(train_x.size()[0]) * 0.2
    return(train_y)
def f_target(x):
    return (6 * x - 2)**2 * torch.sin(12 * x - 4)
#train_y = f_target(train_x).squeeze()
#f_target = target_y
#import ipdb; ipdb.set_trace()
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))#+gpytorch.kernels.WhiteNoiseKernel(variances=torch.ones(train_x.shape))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())#+gpytorch.kernels.WhiteNoiseKernel(variances=torch.ones(train_x.shape))
        #self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)#+gpytorch.kernels.WhiteNoiseKernel(variances=torch.ones(train_x.shape))
        
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TrainedGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(TrainedGPModel, self).__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.train_x = train_x
        self.train_y = train_y

    def train_model(self):
        # initialize likelihood and model
        #import ipdb; ipdb.set_trace()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.raw_noise = -10
        self.likelihood.raw_noise.requires_grad = False
        
        #import ipdb; ipdb.set_trace()
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 501
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            if i % 50 == 0:
                print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.log_lengthscale.item(),
                    self.model.likelihood.log_noise.item()
                ))
            optimizer.step()

    def re_train_model(self):
        # Find optimal model hyperparameters
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 501
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            if i % 50 == 0:
                print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.log_lengthscale.item(),
                    self.model.likelihood.log_noise.item()
                ))
            optimizer.step()

    def update_gp(self, new_x, new_y):
        # update the gp with new points and retrains the model
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        # remove points that have been sampled twice
        X_new, ind_to_remove = remove_close_points(self.train_x.detach().numpy())
        y_new = np.delete(self.train_y.detach().numpy(), ind_to_remove)
        self.train_x = torch.tensor(X_new)
        self.train_y = torch.tensor(y_new)
        self.re_train_model()

    def pred_model(self, test_x):
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        #with torch.no_grad(), gpytorch.fast_pred_var():
        observed_pred = self.likelihood(self.model(test_x))
        f_preds = self.model(test_x)
        self.f_mean = f_preds.mean
        self.f_var = f_preds.variance
        self.f_covar = f_preds.covariance_matrix
        #import ipdb; ipdb.set_trace()
        #self.f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
        return self.f_mean, self.f_covar, observed_pred
        

    def prepare_grad(self, x_q):
        """
            function that helps prepare for the optimization
        """
        self.x_q = torch.tensor(x_q, requires_grad=True)

    def prepare_mu(self, x_mu):
        """
            function that helps prepare for the optimization of the knowledge gradient
        """
        self.x_mu = torch.tensor(x_mu, requires_grad=True)


    def pred_model_grad(self):
        #raise ValueError('dont use this function any more, only for testing the autograd')
        self.model.eval()
        self.likelihood.eval()
        #observed_pred = self.likelihood(self.model(test_x))
        f_preds = self.model(self.x_q)
        f_mean = f_preds.mean
        return(f_mean)


    def q_expected_improvement(self,
        sampling_type='MC', 
        sample_size=200):
        mu_q, variance_q, __ = self.pred_model(self.x_q)
        f_star = (self.train_y).min()
        if sampling_type == 'MC':
            z_sample = torch.normal(torch.zeros(mu_q.shape[0], sample_size))
        elif sampling_type == 'RQMC':
            z_sample = sobol_sequence(sample_size, mu_q.shape[0], IFLAG=1, iSEED=np.random.randint(10**5), TRANSFORM=1).transpose()
            z_sample = torch.tensor(z_sample, dtype=torch.float32, requires_grad=False)
        #sigma = torch.cholesky(variance_q)
        try:
            #sigma = torch.cholesky(variance_q)
            sigma = gpytorch.root_decomposition(variance_q).evaluate()
        except:
            small_epsilon = 0.01
            identity_mat = torch.eye(variance_q.shape[0])
            try:
                #U, S, V = torch.svd(variance_q+small_epsilon*identity_mat)
                #sigma = torch.mm(U, torch.diag(S**0.5))
                #sigma = torch.cholesky(variance_q+small_epsilon*identity_mat)
                sigma = gpytorch.root_decomposition(variance_q+small_epsilon*identity_mat).evaluate()
            except:
                import ipdb; ipdb.set_trace()
                # in case it is still not working
        f_sample = mu_q.unsqueeze(1)+torch.mm(sigma, z_sample)
        #import ipdb; ipdb.set_trace()
        #return -torch.clamp(f_star-f_sample.min(0),0).mean(1).unsqueeze(0)
        #return self.x_q**2
        return -torch.clamp(f_star-f_sample.min(0)[0],0).mean().unsqueeze(0)

    def q_knowledge_gradient(self,
        sampling_type='MC', 
        sample_size=200):
        

        # 1. minimize the mu_n 
        x_init = self.train_x[self.train_y.argmin(),:]
        x_start = x_init.clone().detach().requires_grad_(True).unsqueeze(0)
        self.prepare_mu(x_start)
        #import ipdb; ipdb.set_trace()
        mu_q, __, __ = self.pred_model(self.x_mu)
        #raise ValueError('not implemented yet!')
        optimizer = torch.optim.Adam([self.x_mu], lr=0.01)
        for t in range(10):
            optimizer.zero_grad()
            mu_q, __, __ = self.pred_model(self.x_mu)
            loss = mu_q
            #loss.backward()#retain_grad()
            loss.backward()
            #import ipdb; ipdb.set_trace()
            optimizer.step()
        # results of the first minimization
        x_star = torch.clamp(self.x_mu.clone().detach(), 0., 1.)
        mu_q, __, __ = self.pred_model(x_star)
        mu_star = mu_q.clone().detach()

        import ipdb; ipdb.set_trace()

        # 2. calc expectation
        mu_q, variance_q, __ = self.pred_model(self.x_q)
        if sampling_type == 'MC':
            z_sample = torch.normal(torch.zeros(mu_q.shape[0], sample_size))
        elif sampling_type == 'RQMC':
            z_sample = sobol_sequence(sample_size, mu_q.shape[0], IFLAG=1, iSEED=np.random.randint(10**5), TRANSFORM=1).transpose()
            z_sample = torch.tensor(z_sample, dtype=torch.float32, requires_grad=False)
        #sigma = torch.cholesky(variance_q)
        try:
            #sigma = torch.cholesky(variance_q)
            sigma = gpytorch.root_decomposition(variance_q).evaluate()
        except:
            small_epsilon = 0.01
            identity_mat = torch.eye(variance_q.shape[0])
            try:
                #U, S, V = torch.svd(variance_q+small_epsilon*identity_mat)
                #sigma = torch.mm(U, torch.diag(S**0.5))
                #sigma = torch.cholesky(variance_q+small_epsilon*identity_mat)
                sigma = gpytorch.root_decomposition(variance_q+small_epsilon*identity_mat).evaluate()
            except:
                import ipdb; ipdb.set_trace()
                # in case it is still not working
        f_sample = mu_q.unsqueeze(1)+torch.mm(sigma, z_sample)
        #import ipdb; ipdb.set_trace()
        #return -torch.clamp(f_star-f_sample.min(0),0).mean(1).unsqueeze(0)
        #return self.x_q**2
        return -torch.clamp(f_star-f_sample.min(0)[0],0).mean().unsqueeze(0)


    def analytic_expected_improvement(self):
        xi = 0.00
        mu_q, variance_q, __ = self.pred_model(self.x_q)
        f_star = (self.train_y).min()
        Z = (f_star-mu_q+xi)/(variance_q**0.5)
        normal_dist = torch.distributions.normal.Normal(0,1)
        ei = (f_star-mu_q+xi)*normal_dist.cdf(Z)+(variance_q**0.5)*torch.exp(normal_dist.log_prob(Z))
        #import ipdb; ipdb.set_trace()
        return -ei

#import ipdb; ipdb.set_trace()

def remove_close_points(x_init, dist=.001):
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


def variance_gradient(trained_gp, x_one, m_rep=20, sample_size=200, sampling_type='MC'):
    """
    function that extracts the variance of the gradient
    """
    grad_list = []
    f_list = []
    for m in range(m_rep):
        trained_gp.prepare_grad(x_one)
        loss = trained_gp.q_expected_improvement(sample_size=sample_size, sampling_type=sampling_type)
        f_list.append(loss.detach().numpy())
        loss.backward(retain_graph=True)
        grad_list.append(trained_gp.x_q.grad.data.detach().numpy())
    
    return(np.array(f_list), np.array(grad_list))


def optimize_acquisition(trained_gp, 
    x_start, 
    sampling_type="RQMC", 
    sample_size=20,
    bounds=(0,1), 
    acquisition='qKG'):
    trained_gp.prepare_grad(x_start)
    optimizer = torch.optim.Adam([trained_gp.x_q], lr=0.1)
    for t in range(50):
        optimizer.zero_grad()
        if acquisition == 'qEI':
            loss = trained_gp.q_expected_improvement(sampling_type=sampling_type, sample_size=sample_size)
        elif acquisition == 'qKG':
            loss = trained_gp.q_knowledge_gradient(sampling_type=sampling_type, sample_size=sample_size)
        #loss.retain_grad()
        loss.backward(retain_graph=True)
        #import ipdb; ipdb.set_trace()
        optimizer.step()
    res_x = trained_gp.x_q.detach() 
    return(torch.clamp(res_x, bounds[0], bounds[1]))


def next_x(trained_gp, 
    num_candidates=20, 
    sampling_type="RQMC", 
    sample_size=200,
    q_size=5):
    """
    function that uses a multistart approach in order 
    to find the maximizer of the q-EI
    """
    candidates = []
    values = []
    #import ipdb; ipdb.set_trace()
    dim = trained_gp.train_x.shape[1]
    # TODO: find a way to get a latin hypercube desing with respect to number of points, 
    # dimension of X and q points
        
    for i in range(num_candidates):
        x_init = lhs(n=dim, samples=q_size, criterion='maximin')
        x_init_points = torch.tensor(x_init, dtype=torch.float)
        x = optimize_acquisition(trained_gp, x_init_points, sampling_type=sampling_type, sample_size=sample_size)
        y = trained_gp.q_expected_improvement(sampling_type=sampling_type, sample_size=sample_size)
        
        candidates.append(x)
        values.append(y)
        #import ipdb; ipdb.set_trace()
    argmin = torch.min(torch.cat(values), dim=0)[1].item()
    return candidates[argmin]


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


def run_bo_gpytorch(params_bo, params_data, outer_loop_steps=2, q_size=2):
    """
    run the bo 
    """
    X = params_data['X']
    y = params_data['y']
    #import ipdb; ipdb.set_trace()
    trained_gp = TrainedGPModel(X, y)


    sampling_type = params_bo['sampling_type']
    sample_size = params_bo['sample_size']
    f_target = params_data['f_target']
    print('run model with %s and %s samples' % (sampling_type, sample_size))
    start_time = time.time()
    trained_gp.train_model()

    y_list_min  = []
    x_list_min  = []
    gradient_estimators_list = []
    acquisition_function_estimators_list = []

    for i in range(outer_loop_steps):
        print('approach %s, sample size %s, outer loop step %s of %s'% (sampling_type, sample_size, i, outer_loop_steps))
        xmin = next_x(trained_gp, sampling_type=sampling_type, sample_size=sample_size, q_size=q_size)
        print("next points evaluated:")
        #import ipdb; ipdb.set_trace()
        y = f_target(xmin).squeeze()
        print(xmin, y)
        # update the posterior with the new point found
        #TODO: do we evaluate the target function at all points??
        trained_gp.update_gp(xmin, y)
        # add to the list of best sampled points
        val_y, ind = trained_gp.train_y.min(0)
        val_x = trained_gp.train_x[ind,:]
        y_list_min.append(val_y.detach().numpy())
        x_list_min.append(val_x.detach().numpy())
        print("best points so far:")
        print(x_list_min[-1], y_list_min[-1])

        f_list, grad_list = variance_gradient(trained_gp, xmin, sample_size=sample_size, sampling_type=sampling_type)
        gradient_estimators_list.append(grad_list)
        acquisition_function_estimators_list.append(f_list)


    print("run time %s seconds" %(time.time() -start_time))
    res_dict = {'X_exp' : trained_gp.train_x.detach().numpy(), 'y_exp' :trained_gp.train_y.numpy(),
                'X_min': np.array(x_list_min), 'y_min' : np.array(y_list_min),
                'aquisition_estimation' : acquisition_function_estimators_list,
                'grad_aquisition_estimation' : gradient_estimators_list
                }
    return trained_gp, res_dict

def plot_gp_1d(trained_gp):
    test_x = torch.linspace(0, 1, 100).unsqueeze(1)
    #import ipdb; ipdb.set_trace()
    with torch.no_grad():
        # Initialize plot
        ei_list = []
        aei_list = []
        test_x = torch.linspace(0,1,100).unsqueeze(1)
        #import ipdb; ipdb.set_trace()
        for x in test_x:
            trained_gp.prepare_grad(x)
            ei_list.append(trained_gp.q_expected_improvement().detach().numpy())
            aei_list.append(trained_gp.analytic_expected_improvement().detach().numpy())
        #import ipdb; ipdb.set_trace()
        f_mean, f_covar, observed_pred = trained_gp.pred_model(test_x)
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(trained_gp.train_x.numpy(), trained_gp.train_y.numpy(), 'k*')

        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.plot(test_x.numpy(), 100*np.array(ei_list).flatten(), 'r')
        ax.plot(test_x.numpy(), 100*np.array(aei_list).flatten(), 'g')
        #ax.plot(trained_gp.x_q.detach().numpy(), trained_gp.pred_model_grad().detach().numpy(), 'k*', color="green")
        
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy().flatten(), lower.numpy(), upper.numpy(), alpha=0.5)

        #ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'ei', 'ai', 'Confidence'])
        plt.show()


if __name__ == '__main__':
    dim = 1
    np.random.seed(42)
    X = np.random.random(size=(5, dim))
    X = torch.tensor(X, dtype=torch.float)
    y = f_target(X).squeeze()

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
    res_model_rqmc, rqmc_dict = run_bo_gpytorch(params_bo_rqmc, params_data)
    plot_gp_1d(res_model_rqmc)
    import ipdb; ipdb.set_trace()



