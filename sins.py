import numpy as np
import matplotlib.pyplot as plt
import nevergrad as ng

import csv
import os
from functools import partial

import tqdm
from concurrent import futures

def sinusoid(x, minn, maxx, phase):
    amp = (maxx - minn) / 2
    offset = (maxx + minn) / 2
    return amp * np.sin(2 * np.pi * x + phase) + offset

def objective(params, stack, stack_ind, cstack_size, n_angles, n_times):
    minn1, maxx1, phase1, minn2, maxx2, phase2 = params.values()
    assert isinstance(minn1, float) and isinstance(maxx1, float) and isinstance(phase1, float) and isinstance(minn2, float) and isinstance(maxx2, float) and isinstance(phase2, float), "Params must be floats but are of type: " + str([type(param) for param in params])
    minns, maxxs, phases = np.linspace(minn1, minn2, cstack_size), np.linspace(maxx1, maxx2, cstack_size), np.linspace(phase1, phase2, cstack_size)
    x = np.arange(n_angles) / n_angles
    cumsum = 0
    for i in range(cstack_size):
        y = sinusoid(x, minns[i], maxxs[i], phases[i])
        y = ((n_times-1) * y).astype(int) # Scale y indices to image height
        pixel_values = stack[:, i][np.arange(n_angles), y]  # Get pixel values along the sinusoid
        gain = pixel_values.sum() 
        log_results(stack_ind+i, minns[i], maxxs[i], phases[i], gain)
        cumsum += gain
    return -cumsum  # Minimize the negative sum to maximize

# Define a simple logger function
def log_results(row, minn, maxx, phase, loss, filepath="optimization_log.csv"):
    # Check if the file exists to write headers
    file_exists = os.path.isfile(filepath)
    res_dict = {'row': row, 'minn': minn, 'maxx': maxx, 'phase': phase, 'loss': loss}
    with open(filepath, "a", newline="") as csvfile:
        fieldnames = list(res_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write headers if file does not exist
        
        writer.writerow(res_dict)  # Write the results to the file


def main(stack_size=20):
    """Now we compute the best sinusoids for the last row, and then use this as initialization to go to the previous row and so on."""
    terrains = np.load('terrains.npy')  # A, R, T  (angles, rows, time)
    # clean those angles that are not very correlated with the others
    corrs = np.einsum('art,brt->abt', terrains, terrains)
    corrs = np.mean(corrs, axis=2).sum(axis=0)  # A
    hist, bin_edges = np.histogram(corrs, bins=10)
    for stack_ind in range(len(hist) - 1, -1, -1):  # not very efficient
        if hist[stack_ind] == 0:
            break
    corr_thresh = bin_edges[stack_ind - 1]
    bad_row_indices = corrs < corr_thresh
    terrains[bad_row_indices] = terrains[terrains > 0].mean()

    if os.path.exists("optimization_log.csv"):
        os.remove("optimization_log.csv")

    # now we optimize stacks of sinusoids
    for stack_ind in tqdm.tqdm(range(terrains.shape[1] - stack_size)):
        stack = terrains[:, stack_ind:stack_ind+stack_size]  # (A, stack_size, T)
        cstack_size = stack.shape[1]
        n_angles = stack.shape[0]
        n_times = stack.shape[2]




        param_dict = ng.p.Dict(minn1=ng.p.Scalar(lower=0, upper=1), maxx1=ng.p.Scalar(lower=0, upper=1), phase1=ng.p.Scalar(lower=0, upper=np.pi * 2), minn2=ng.p.Scalar(lower=0, upper=1), maxx2=ng.p.Scalar(lower=0, upper=1), phase2=ng.p.Scalar(lower=0, upper=np.pi * 2))
        def constraint(d):
            return (d['minn1'] < d['maxx1']) and (d['minn2'] < d['maxx2'])
        param_dict.register_cheap_constraint(constraint)
        obj = partial(objective, stack=stack, stack_ind=stack_ind, cstack_size=cstack_size, n_angles=n_angles, n_times=n_times)
        # optimizer = ng.optimizers.NGOpt(parametrization=param_dict, budget=256, num_workers=8, )
        optimizer = ng.optimizers.PSO(parametrization=param_dict, budget=32, num_workers=8)
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(obj, verbosity=0, executor=executor, batch_mode=False)

    
    # plt.figure()
    # plt.title(objective(*optimized_params))
    # plt.imshow(image_slice1)
    # plt.plot(sinusoid(np.arange(n_angles) / n_angles, *optimized_params) * (n_times-1), np.arange(n_angles), 'r')
    # plt.show()

    # while True:
    #     minn = float(input('Enter minn: ')); print()
    #     maxx = float(input('Enter maxx: ')); print()
    #     phase = float(input('Enter phase: ')); print()
    #     params = [minn, maxx, phase]
    #     plt.figure()
    #     plt.title(objective(*params))
    #     plt.imshow(image_slice1)
    #     plt.plot(sinusoid(np.arange(n_angles) / n_angles, *params) * (n_times-1), np.arange(n_angles), 'r')
    #     plt.show()
    #     breakpoint()

    # result = minimize(objective, initial_params, bounds=bounds, method='Nelder-Mead', options={'maxiter': 10000})


    # if result.success:
    #     optimized_params = result.x
    #     print("Optimized Parameters:", optimized_params)
    #     plt.figure()
    #     plt.title(objective(optimized_params))
    #     plt.imshow(image_slice)
    #     plt.plot(sinusoid(np.arange(n_angles) / n_angles, *optimized_params) * (n_times-1), np.arange(n_angles), 'r')
    #     plt.show()
 
    # else:
    #     print("Optimization was not successful.")


if __name__ == '__main__':
    main()