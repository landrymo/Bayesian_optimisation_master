# main simulation based on pyro
import pickle
import torch
#   import torch.multiprocessing as mp
#import pathos.multiprocessing as mp
import numpy as np
import sys
from functools import partial
#from concurrent.futures import ProcessPoolExecutor

from fit_gp import run_bo_gpytorch, random_search
from target_functions import f_hart6, f_branin, f_threehump, f_schaffer2, f_rosenbrock

target_functions_dict = {
    #'f_hart6' : {'f_target' : f_hart6, 'dim' : 6},
    'f_branin' : {'f_target' : f_branin, 'dim' : 2},
    'f_threehump' : {'f_target' : f_threehump, 'dim' : 2},
    'f_schaffer2' : {'f_target' : f_schaffer2, 'dim' : 2},
    'f_rosenbrock' : {'f_target' : f_rosenbrock, 'dim' : 4}
}


q_size_list = [2, 5, 10]
sample_sizes_list = [50, 100, 200, 500]


def parallel_calc(m_rep_iter, params_bo_mc, params_bo_rqmc, params_data, outer_loop_steps=5, q_size=2):
    __, random_search_dict = random_search(params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)
    __, mc_dict = run_bo_gpytorch(params_bo_mc, params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)
    __, rqmc_dict = run_bo_gpytorch(params_bo_rqmc, params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)
    return [mc_dict, rqmc_dict, random_search_dict]

def loop_over_parameters(dict_targets, sample_sizes_list, q_size_list, m_rep_iter, Mrep = 20, outer_loop_steps = 2):
    """
    function that loop over the different parameter settings
    """
    for key, value in dict_targets.items():
        print('now runinning %s' % key)
        f_target = value['f_target']
        dim = value['dim']
        np.random.seed(42)
        X = np.random.random(size=(5, dim)) # using 5 starting points
        X = torch.tensor(X, dtype=torch.float)
        y = f_target(X)

        np.random.seed(m_rep_iter)


        params_data = {
            'X' : X,
            'y' : y,
            'dim' : dim,
            'f_target' : f_target
            }

        params_bo_mc = {
            'sampling_type' : 'MC', 
            'sample_size' : sample_sizes_list[0],
            'num_candidates' : 10
        }

        params_bo_rqmc = {
            'sampling_type' : 'RQMC', 
            'sample_size' : sample_sizes_list[0],
            'num_candidates' : 10
        }
        res_dict = {'MC': {str(sample_size): [] for sample_size in sample_sizes_list}, 
                    'RQMC' : {str(sample_size): [] for sample_size in sample_sizes_list},
                    'random_search' : {str(sample_size): [] for sample_size in sample_sizes_list}
                    }
        for q_size in q_size_list:
            for sample_size in sample_sizes_list:
                params_bo_mc['sample_size'] = sample_size
                params_bo_rqmc['sample_size'] = sample_size
                partial_parallel_calc = partial(parallel_calc, params_bo_mc=params_bo_mc, params_bo_rqmc=params_bo_rqmc, params_data=params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)

                mc_dict, rqmc_dict, random_search_dict = partial_parallel_calc(m_rep_iter)
                #__, random_search_dict = random_search(params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)
                #__, mc_dict = run_bo_gpytorch(params_bo_mc, params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)
                #__, rqmc_dict = run_bo_gpytorch(params_bo_rqmc, params_data, outer_loop_steps=outer_loop_steps, q_size=q_size)
                res_dict['MC'][str(sample_size)].append(mc_dict)
                res_dict['RQMC'][str(sample_size)].append(rqmc_dict)
                res_dict['random_search'][str(sample_size)].append(random_search_dict)
                
                with open('pyro_bo_mrep_%s_%s_%s_q_size_%s.pkl'%(m_rep_iter, Mrep, f_target.__name__, q_size), 'wb') as file:
                    pickle.dump(res_dict, file, protocol=2)


if __name__ == '__main__':
    Mrep_total = 20
    
    if len(sys.argv)>1:
        m_rep = int(sys.argv[1])
        loop_over_parameters(target_functions_dict, sample_sizes_list, q_size_list, m_rep, Mrep = Mrep_total, outer_loop_steps = 25)
        import ipdb; ipdb.set_trace()
    else:
        for m_rep in range(Mrep_total):
            loop_over_parameters(target_functions_dict, sample_sizes_list, q_size_list, m_rep, Mrep = Mrep_total, outer_loop_steps = 25)