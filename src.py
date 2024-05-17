import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from numba import njit


def swap_process(graph, return_cut=False):
    """
    not the most efficient, but at this point i don't care
    makes a random cut, while there are unhappy vertices switches them
    returns true if got an internal cut, false if everybody is on one side
    """
    
    num_vxs = graph.number_of_nodes()
    adj_matrix = nx.to_numpy_array(graph, nodelist=range(num_vxs))
    
    cut = np.random.choice([-1, 1], size=num_vxs)
    balances = (adj_matrix @ cut) * cut
    unbalanced_indices = (balances < 0).nonzero()[0]

    while unbalanced_indices.size != 0:
        swap_index = np.random.choice(unbalanced_indices)
        cut[swap_index] *= -1
        # balances = (adj_matrix @ cut) * cut
        balances[swap_index] *= -1
        if cut[swap_index] == 1:
            balances += adj_matrix[:, swap_index] * 2 * cut
        else:
            balances -= adj_matrix[:, swap_index] * 2 * cut
        
        unbalanced_indices = (balances < 0).nonzero()[0]

    if return_cut:
        return (not(np.all(cut == 1) or np.all(cut == -1))), cut
    return not(np.all(cut == 1) or np.all(cut == -1))


def measure_success_probability(d, num_vxs, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        if (swap_process(graph)):
            num_successes += 1

    return num_successes / num_tests


def measure_erdos_renyi(p, num_vxs, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.erdos_renyi_graph(num_vxs, p)
        if (swap_process(graph)):
            num_successes += 1

    return num_successes / num_tests


def found_partition(graph, num_tries):
    for i in range(num_tries):
        if swap_process(graph):
            return True
    return False


def produce_counterexample(d, num_vxs, iterations_thresh=1000, modifications_thresh=1000, silent=True):
    graph = nx.random_regular_graph(d, num_vxs)

    for i in range(modifications_thresh):
        if i % 100 == 0 and not silent:
            print(i)
        
        result, cut = swap_process(graph, return_cut=True)
        iteration = 0
        while (not result) and iteration < iterations_thresh:
            result, cut = swap_process(graph, return_cut=True)
            iteration += 1

        if not result:
            print(f'found potential counterexample at iteration {i}')
            return True, graph
        
        edge_pos = random.choice(list(graph.edges()))
        while cut[edge_pos[0]] != 1 or cut[edge_pos[1]] != 1:
            edge_pos = random.choice(list(graph.edges()))
        
        edge_neg = random.choice(list(graph.edges()))
        while cut[edge_neg[0]] != -1 or cut[edge_neg[1]] != -1:
            edge_neg = random.choice(list(graph.edges()))

        if not graph.has_edge(edge_pos[0], edge_neg[0]) and not graph.has_edge(edge_pos[1], edge_neg[1]):
            graph.remove_edges_from([edge_pos, edge_neg])
            graph.add_edges_from([(edge_pos[0], edge_neg[0]), (edge_pos[1], edge_neg[1])])

    return False, graph


def majority_dynamics_step(graph, cut):
    num_vxs = graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(graph, nodelist=range(num_vxs))
    
    balances = (adj_matrix @ cut) * cut
    unbalanced_indices = (balances < 0).nonzero()[0]

    cut_copy = cut.copy() # don't want to change the cut itself
    cut_copy[unbalanced_indices] *= -1
    return cut_copy


def is_core_in_positive_part(graph, cut, return_core=False):
    """
    returns if there is a subgraph inside the positive side of the cut
    with min deg >= d/2
    """
    num_vxs = graph.number_of_nodes()
    adj_matrix = nx.to_numpy_array(graph, nodelist=range(num_vxs))

    balances = (adj_matrix @ cut) * cut
    unbalanced_indices = ((balances < 0) * (cut > 0)).nonzero()[0]
    while unbalanced_indices.size != 0:
        swap_index = np.random.choice(unbalanced_indices)
        cut[swap_index] = -1
        balances[swap_index] *= -1
        balances -= adj_matrix[:, swap_index] * 2 * cut
        unbalanced_indices = ((balances < 0) * (cut > 0)).nonzero()[0]

    if not return_core:
        return np.any(cut == 1)
    return np.any(cut == 1), cut


def core_probability(d, num_vxs, num_dyn_steps, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        for j in range(num_dyn_steps):
            cut = majority_dynamics_step(graph, cut)
            
        if is_core_in_positive_part(graph, cut):
            num_successes += 1

    return num_successes / num_tests


def run_swap_process(graph, cut, num_steps, return_cut=False):
    num_vxs = graph.number_of_nodes()
    adj_matrix = nx.to_numpy_array(graph, nodelist=range(num_vxs))
    
    balances = (adj_matrix @ cut) * cut
    unbalanced_indices = (balances < 0).nonzero()[0]

    step = 0
    while unbalanced_indices.size != 0 and step < num_steps:
        step += 1
        swap_index = np.random.choice(unbalanced_indices)
        cut[swap_index] *= -1
        # balances = (adj_matrix @ cut) * cut
        balances[swap_index] *= -1
        if cut[swap_index] == 1:
            balances += adj_matrix[:, swap_index] * 2 * cut
        else:
            balances -= adj_matrix[:, swap_index] * 2 * cut
        
        unbalanced_indices = (balances < 0).nonzero()[0]

    if return_cut:
        return (not(np.all(cut == 1) or np.all(cut == -1))), cut
    return not(np.all(cut == 1) or np.all(cut == -1))


def swap_core_probability(d, num_vxs, num_steps, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        _, cut = run_swap_process(graph, cut, num_steps, return_cut=True)
            
        if is_core_in_positive_part(graph, cut):
            num_successes += 1

    return num_successes / num_tests


def clip(array):
    """
    array: 1d numpy array
    for each element e of array,
    if a > 1 make a = 1
    if a < -1 make a = -1
    """
    array = np.min(np.vstack((array, np.ones_like(array))), axis=0)
    array = np.max(np.vstack((array, - np.ones_like(array))), axis=0)
    return array.astype(int)


def majority_dynamics_step_with_zeros(graph, cut):
    """
    cut[i] is in {-1, 0, 1}
    if the sum of neighbors is positive(negative), then move towards that side (through zero)
    """
    
    num_vxs = graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(graph, nodelist=range(num_vxs))
    return clip(cut + clip(adj_matrix @ cut))


def core_probability_with_zeros(d, num_vxs, num_dyn_steps, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        for j in range(num_dyn_steps):
            cut = majority_dynamics_step_with_zeros(graph, cut)
        cut[cut == 0] = -1
        
        if is_core_in_positive_part(graph, cut):
            num_successes += 1

    return num_successes / num_tests


def zeros_fraction_with_zeros(d, num_vxs, num_dyn_steps, num_tests, silent=True):
    num_zeros = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        for j in range(num_dyn_steps):
            cut = majority_dynamics_step_with_zeros(graph, cut)
        
        num_zeros += np.count_nonzero(cut==0)

    return num_zeros / num_tests / num_vxs


def changing_fraction_md(d, num_vxs, num_dyn_steps, num_tests, silent=True):
    changed = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        for j in range(num_dyn_steps):
            cut = majority_dynamics_step(graph, cut)
        cut_copy = cut.copy()
        cut = majority_dynamics_step(graph, cut)
        
        changed += np.count_nonzero(cut - cut_copy)

    return changed / num_tests / num_vxs


def proba_oscillations_md(d, num_vxs, num_dyn_steps, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        for j in range(num_dyn_steps):
            cut = majority_dynamics_step(graph, cut)
        cut_copy = cut.copy()
        cut = majority_dynamics_step(graph, cut)
        cut = majority_dynamics_step(graph, cut)
        
        if np.array_equal(cut, cut_copy):
            num_successes += 1

    return num_successes / num_tests


def majority_dynamics_step_in_halves(graph, cut):
    num_vxs = graph.number_of_nodes()
    adj_matrix = nx.to_numpy_array(graph, nodelist=range(num_vxs))

    cut = clip(cut + 2 * np.max(np.vstack((np.zeros_like(cut), adj_matrix @ cut)), axis=0))
    cut = clip(cut + 2 * np.min(np.vstack((np.zeros_like(cut), adj_matrix @ cut)), axis=0))
    return cut


def core_probability_in_halves(d, num_vxs, num_dyn_steps, num_tests, silent=True):
    num_successes_positive = 0
    num_successes_negative = 0

    for i in range(num_tests):
        if i % 1000 == 0 and not silent:
            print(f'i: {i}, num_successes: {num_successes}')

        graph = nx.random_regular_graph(d, num_vxs)
        
        cut = np.random.choice([1, -1], size=num_vxs)
        for j in range(num_dyn_steps):
            cut = majority_dynamics_step_in_halves(graph, cut)
        
        if is_core_in_positive_part(graph, np.copy(cut)):
            num_successes_positive += 1
        if is_core_in_positive_part(graph, - np.copy(cut)):
            num_successes_negative += 1


    return num_successes_positive / num_tests, num_successes_negative / num_tests


def generate_result(graph):
    # runs Majority Dynamics until period 2 is reached,
    # returns two consequtive states of the cut after that
    
    num_vxs = graph.number_of_nodes()

    cut = np.random.choice([1, -1], size=num_vxs)
    cut_copy = cut.copy()
    cut = majority_dynamics_step(graph, cut)
    cut = majority_dynamics_step(graph, cut)

    while not np.array_equal(cut, cut_copy):
        cut_copy = cut.copy()
        cut = majority_dynamics_step(graph, cut)
        cut = majority_dynamics_step(graph, cut)

    return cut_copy, majority_dynamics_step(graph, cut)


def draw_result(graph, result, labels=False):
    colors = ['red' if x == 1 else 'blue' for x in result[0]]
    edgecolors = ['black' if x != y else 'white' for x, y in zip(result[0], result[1])]
    nx.draw_networkx(graph, 
                     node_color=colors, 
                     with_labels=labels, 
                     pos=nx.spring_layout(graph), 
                     edgecolors=edgecolors, 
                     linewidths=3, 
                     nodelist=range(graph.number_of_nodes()))
    plt.show()


def oscillating_fractions(num_vxs, d, num_tests):
    ans = []
    for i in range(num_tests):
        graph = nx.random_regular_graph(d, num_vxs)
        result = generate_result(graph)
        ans.append(np.sum(result[0] != result[1]) / num_vxs)

    return ans
    

def steps_until_converge(graph):
    # runs Majority Dynamics until period 2 is reached,
    # returns the number of steps until a state in the final cycle is reached for the first time
    
    num_vxs = graph.number_of_nodes()

    cut = np.random.choice([1, -1], size=num_vxs)
    next_cut = majority_dynamics_step(graph, cut)
    next_next_cut = majority_dynamics_step(graph, next_cut)

    steps = 0

    while not np.array_equal(cut, next_next_cut):
        steps += 1
        cut = next_cut
        next_cut = next_next_cut
        next_next_cut = majority_dynamics_step(graph, next_next_cut)

    return steps

def oscillating_fractions_er(num_vxs, p, num_tests):
    ans = []
    for i in range(num_tests):
        graph = nx.erdos_renyi_graph(num_vxs, p)
        result = generate_result(graph)
        ans.append(np.sum(result[0] != result[1]) / num_vxs)

    return ans

def steps_until_approximately_converge(graph, fraction):
    # runs Majority Dynamics until between x_t and x_{t+2} the fraction of differces is at most fraction,
    # returns the number of steps t
    
    num_vxs = graph.number_of_nodes()

    cut = np.random.choice([1, -1], size=num_vxs)
    next_cut = majority_dynamics_step(graph, cut)
    next_next_cut = majority_dynamics_step(graph, next_cut)

    steps = 0

    while np.sum(cut == next_next_cut) / num_vxs < fraction:
        steps += 1
        cut = next_cut
        next_cut = next_next_cut
        next_next_cut = majority_dynamics_step(graph, next_next_cut)

    return steps


def core_probability_final(d, num_vxs, num_tests, silent=True):
    num_successes = 0

    for i in range(num_tests):
        graph = nx.random_regular_graph(d, num_vxs)
        cut, _ = generate_result(graph)
        if is_core_in_positive_part(graph, cut):
            num_successes += 1

    return num_successes / num_tests
