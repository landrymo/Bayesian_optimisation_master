# target functions
import numpy as np
import torch

def f_norm(x):
    """
    the target target function that we are optimizing
    """
    return torch.norm((x -0.5), dim=1)
    

def f_hart6(x,
          alpha=np.asarray([1.0, 1.2, 3.0, 3.2]),
          P=10**-4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]]),
          A=np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])):
    """The six dimensional Hartmann function is defined on the unit hypercube.

    It has six local minima and one global minimum f(x*) = -3.32237 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

    More details: <http://www.sfu.ca/~ssurjano/hart6.html>
    """
    DIM = 6
    assert x.shape[1] == DIM # check the correct size
    x_multiplicity =  x.shape[0]
    res_list = []
    for i in range(x_multiplicity):
        res = -np.sum(alpha * np.exp(-np.sum(A * (np.array(x[i,:]) - P)**2, axis=1)))
        res_list.append(res)
    #import ipdb; ipdb.set_trace()
    return torch.tensor(res_list, dtype=torch.float)


def f_branin(x, a=1, b=5.1 / (4 * np.pi**2), c=5. / np.pi,
           r=6, s=10, t=1. / (8 * np.pi)):
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: <http://www.sfu.ca/~ssurjano/branin.html>
    """
    #import ipdb; ipdb.set_trace()
    assert x.shape[1] == 2
    # transform the samples, since we restrict the space to [0,1]
    x0 = (x[:,0]*15)-5 # x1 ∈ [-5, 10]
    x1 = x[:,1]*15 # x2 ∈ [0, 15]
    return (a * (x1 - b * x0 ** 2 + c * x0 - r) ** 2 +
            s * (1 - t) * np.cos(x0) + s)

def f_rastrigin(x):
    """Rastrigin objective function.
    https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/utils/functions/single_obj.html
    Has a global minimum at :code:`f(0,0,...,0)` with a search
    domain of :code:`[-5.12, 5.12]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    # transform x to the domain of interest
    x = (x*5.12*2)-5.12
    if not np.logical_and(x >= -5.12, x <= 5.12).all():
        raise ValueError(
            "Input for Rastrigin function must be within " "[-5.12, 5.12]."
        )

    d = x.shape[1]
    j = 10.0 * d + (x ** 2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)

    return j



def f_rosenbrock(x):
    """Rosenbrock objective function.

    Also known as the Rosenbrock's valley or Rosenbrock's banana
    function. Has a global minimum of :code:`np.ones(dimensions)` where
    :code:`dimensions` is :code:`x.shape[1]`. The search domain is
    :code:`[-inf, inf]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    # transform x to the domain of interest
    x = (x*100.*2)-100.
    try:
        r = (100 * (x[:,1:] - x[:,:-1] ** 2.0) ** 2 + (1 - x[:,:-1]) ** 2.0).sum(1)
    except:
        import ipdb; ipdb.set_trace()
    return r



def f_schaffer2(x):
    """Schaffer N.2 objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-100,100]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    # transform x to the domain of interest
    x = (x*100.*2)-100.

    if not x.shape[1] == 2:
        raise IndexError(
            "Schaffer N. 2 function only takes two-dimensional " "input."
        )
    if not np.logical_and(x >= -100, x <= 100).all():
        raise ValueError(
            "Input for Schaffer function must be within " "[-100, 100]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 0.5 + (
        (np.sin(x_ ** 2.0 - y_ ** 2.0) ** 2.0 - 0.5)
        / ((1 + 0.001 * (x_ ** 2.0 + y_ ** 2.0)) ** 2.0)
    )

    return j




def f_threehump(x):
    """Three-hump camel objective function

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([0, 0])`. Its coordinates are bounded within
    :code:`[-5, 5]`.

    Best visualized in the full domin and a range of :code:`[0, 2000]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    # transform x to the domain of interest
    x = (x*5.*2)-5
    if not x.shape[1] == 2:
        raise IndexError(
            "Three-hump camel function only takes two-dimensional input."
        )
    if not np.logical_and(x >= -5, x <= 5).all():
        raise ValueError(
            "Input for Three-hump camel function must be within [-5, 5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = 2 * x_ ** 2 - 1.05 * (x_ ** 4) + (x_ ** 6) / 6 + x_ * y_ + y_ ** 2

    return j