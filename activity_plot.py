import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools 

import sys
decay_rate = sys.argv[1]

def decay_rule(t, rate='fast'):
    if rate == 'fast':
        return 50*np.exp(- t/(num_timesteps/5))
    if rate == 'slow':
        return 50*(1-t/num_timesteps)

num_timesteps= 100

start_states_neurons = [np.random.uniform(0, num_timesteps) for j in range(5)] 
end_states_neurons = [np.random.uniform(0, num_timesteps) for j in range(5)] 
print(start_states_neurons)
print(end_states_neurons)
neuron_number = list(range(5))

triple_combos = list(itertools.combinations(neuron_number, 3)) # find the 5 choose 3 different combinations
fig = plt.figure(figsize=(30, 10))

states_neuron = []
for neuron in  neuron_number:
    mu_start = start_states_neurons[neuron]
    mu_end= end_states_neurons[neuron]
    states_neuron.append([ np.abs( np.random.normal((1-t/num_timesteps)*mu_start + (t/num_timesteps)*mu_end, decay_rule(t, rate=decay_rate)) ) for t in range(num_timesteps) ])
states_neuron = np.array(states_neuron)

counter = 0
for neuron_subset in triple_combos:
    
    plot_states = states_neuron[np.array(neuron_subset), :]
    
    counter += 1
    ax = fig.add_subplot(2, 5, counter, projection ='3d')
    ax.set_xlabel('Neuron {} / HZ'.format(neuron_subset[0]))
    ax.set_ylabel('Neuron {} / HZ'.format(neuron_subset[1]))
    ax.set_zlabel('Neuron {} / HZ'.format(neuron_subset[2]))

    p = ax.scatter3D(plot_states[0,:], 
                    plot_states[1, :], 
                    plot_states[2,:], 
                    c=list(range(plot_states.shape[1])), cmap='plasma')
    fig.colorbar(p)

plt.savefig('activity_output_{}.jpg'.format(decay_rate))


