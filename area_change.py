#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import seaborn as sns
#
# y axis: average change over whole map, in Hz (starts at 50), down to 50
max_variance = 300
variances = np.linspace(0,max_variance,10) # 10 epochs
time_steps = np.linspace(0,999, 50)
#
def f(variance, time): 
    constant = 1000
    initial_magnitude = 50
    return initial_magnitude* (
            (1-variance/max_variance)*( (time/constant-1) ** 4) \
                    + variance/max_variance*( np.sqrt(np.sqrt( 1- time/constant)))
                    )

def noise (variance, time):
    return np.random.normal(0, 3, variance.shape)

def confidence (variance, time):
    pass

var, time = np.meshgrid(variances, time_steps)
delta_activity = f(var, time)
epsilon = noise(var, time)


# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
#===============
#  First subplot
#===============
ax = fig.add_subplot(1, 3, 1)

monkey_data = []
var_data = []
time_data = []
activity_data = []
for this_var in variances: 
    for this_time in time_steps:
        delta_activity_scalar = f(this_var, this_time)
        for monkey in range(14):
            raw_delta = delta_activity_scalar + np.random.normal(0, 5) # 14 monkeys
            monkey_data.append(monkey)
            var_data.append(this_var)
            time_data.append(this_time)
            activity_data.append(raw_delta)
data_dict = {'Monkey Id': monkey_data, 
        'Noise Variance': var_data,
        'Time (ms)': time_data,
        'activity change rate (Hz/s)': activity_data}
dataset = pd.DataFrame(data=data_dict)
sns.lineplot(x='Time (ms)', y='activity change rate (Hz/s)', hue="Noise Variance", 
        data=dataset, markers=True, ax=ax)

#===============
#  Second subplot
#===============
# set up the axes for the first plot
ax = fig.add_subplot(1, 3, 2, projection='3d')

p = ax.contour3D(var, time, delta_activity+epsilon,100 , cmap='plasma')
ax.set_xlabel('variance')
ax.set_ylabel('time (ms)')
ax.set_zlabel('activity_change_rate (Hz/s)');
fig.colorbar(p, shrink=0.5, aspect=10)

#===============
# Third subplot
#===============
# set up the axes for the second plot
ax = fig.add_subplot(1, 3, 3, projection='3d')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
p = ax.contour3D(var, time, delta_activity ,100 , cmap='plasma')
ax.set_xlabel('variance')
ax.set_ylabel('time (ms)')
ax.set_zlabel('activity_change_rate (Hz/s)');
fig.colorbar(p, shrink=0.5, aspect=10)

plt.show()

