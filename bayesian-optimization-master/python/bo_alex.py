import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

import copy
import ipdb

import warnings

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore",category=DeprecationWarning)
import sklearn.gaussian_process as gp

# define the function that we want to minimize
def barnin_function(x):
    """
    barnin function
    input
        x :  a 2 x N np.array; x_1 needs to be in [-5,10], x_2 needs to be in [0,15]
    """
    # fixed values
    a = 1.; b = 5.1/(4*np.pi**2); c = 5./np.pi
    r = 6.; s = 10.; t = 1/(8.*np.pi)
    #x = np.atleast_2d(x)
    x1 = x[0,:]
    x2 = x[1,:]
    assert (-5 <= x1).any()
    assert ( x1 <= 10).any()
    assert (0 <= x2).any()
    assert ( x2 <= 15).any()
    
    res = a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1)+s
    return res 
def hyper_ellipsoid_function(x):
    """

    """
    return (x**2).sum(axis=0)

N = 20
if False: 
    target_function = barnin_function
    x = np.random.random((2,N))*15+np.array([[-5],[0]])
    bounds = np.array([[-5,10], [0,15]])
if True: 
    target_function = hyper_ellipsoid_function
    x = np.random.random((2,N))*10+np.array([[-5],[5]])
    bounds = np.array([[-5,5], [-5,5]])



y = target_function(x)



kernel = gp.kernels.Matern()
model_gp = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=1e-5,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)



# plot the surface
lambdas = np.linspace(-5, 10, 100)
gammas = np.linspace(0, 15, 100)

# We need the cartesian combination of these two vectors
param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])
real_loss = [barnin_function(params[:,np.newaxis]) for params in param_grid]
# The maximum is at:
print param_grid[np.array(real_loss).argmin(), :]

C, G = np.meshgrid(lambdas, gammas)
plt.figure()
cp = plt.contourf(C, G, np.array(real_loss).reshape(C.shape))
plt.colorbar(cp)
plt.savefig('surface_grid.png')
#plt.show()
plt.clf()
# plot the GP surface
model_gp.fit(x.transpose(), y)
real_loss = [model_gp.predict(params.reshape(1,-1)) for params in param_grid]

# The maximum is at:
print param_grid[np.array(real_loss).argmin(), :]
C, G = np.meshgrid(lambdas, gammas)
plt.figure()
cp = plt.contourf(C, G, np.array(real_loss).reshape(C.shape))
plt.colorbar(cp)
plt.savefig('surface_gp.png')
#plt.show()
plt.clf()



def expected_improvement(x, gaussian_process, evaluated_loss):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
    """
    
    x_to_predict = x.reshape(1,-1)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    loss_optimum = np.min(evaluated_loss)


    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = (mu - loss_optimum) / sigma
        expected_improvement = (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return  expected_improvement

def maximize_gp(gaussian_process, bounds, n_restarts=1):
    """
    we maximize the gp in order to use calculate the knowledge gradient
    """
    n_params = bounds.shape[0]
    best_x = None
    best_mu = 1000000
    #ipdb.set_trace()
    target_inter = lambda x: gaussian_process.predict(np.atleast_2d(x))

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        res = minimize(fun=target_inter, x0=starting_point, bounds=bounds, method='L-BFGS-B')


        if res.fun < best_mu:
            best_mu = res.fun
            best_x = res.x
    return best_x, best_mu
    #max_gp = minimize(gaussian_process.predict)
def calculate_kg(x_current, gaussian_process, x, y, bounds):
    """
    calculate the knowledge gradient
    """
    #ipdb.set_trace()
    best_x, best_mu = maximize_gp(gaussian_process, bounds)
    mu_np1_list = []
    for i in range(20):
        y_sampled = gaussian_process.sample_y(best_x.reshape(1,-1), n_samples=1, random_state=i)
        
        y_intermediate = np.hstack((y, y_sampled.flatten()))
        x_intermediate = np.hstack((x, best_x[:,np.newaxis]))

        gp_intermediate = copy.copy(gaussian_process)
        gp_intermediate.fit(x_intermediate.transpose(), y_intermediate)
        __, best_mu_np1 = maximize_gp(gp_intermediate, bounds)
        mu_np1_list.append(best_mu_np1)

    kg = (np.array(mu_np1_list)-best_mu).mean()
    
    
    return(kg)



def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss,
                               bounds=(0, 10), n_restarts=10):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 100000
    n_params = bounds.shape[0]
    #bounds = bounds.transpose()

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        #ipdb.set_trace()
        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

#ipdb.set_trace()

#x_new = np.random.random((2,1))*15+np.array([[-5],[0]])
#expected_improvement(x_new, model_gp, y)
kg_list = []
for i_rep in range(100):
    next_point = sample_next_hyperparameter(expected_improvement, model_gp, y, bounds=bounds, n_restarts=25)
    #ipdb.set_trace()
    next_loss = target_function(next_point[:,np.newaxis])
    #ipdb.set_trace()
    kg_list.append(calculate_kg(next_point, model_gp, x, y, bounds))
    y = np.hstack((y, next_loss))
    x = np.hstack((x, next_point[:,np.newaxis]))
    model_gp.fit(x.transpose(), y)
    
ipdb.set_trace()

real_loss = [model_gp.predict(params.reshape(1,-1)) for params in param_grid]

# The maximum is at:
print param_grid[np.array(real_loss).argmin(), :]
C, G = np.meshgrid(lambdas, gammas)
plt.figure()
cp = plt.contourf(C, G, np.array(real_loss).reshape(C.shape))
plt.scatter(x[0,:], x[1,:])
plt.colorbar(cp)
plt.savefig('surface_gp_end.png')
plt.clf()
ipdb.set_trace()