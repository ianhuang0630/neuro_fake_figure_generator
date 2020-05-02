# import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import numpy as np 
from scipy.stats import multivariate_normal
import seaborn as sns

entropy_level = np.linspace(1, 0.05, 20)
cmap = sns.cubehelix_palette(as_cmap=True)

fig = plt.figure(figsize=(10, 10 ))
counter = 0
#every 50 ms 
for timestep in range(1, 17):
    this_entropy_level = entropy_level[timestep]
    counter += 1
    
    ax = fig.add_subplot(4,4,counter)

    circle = Circle((0,0), 1, facecolor='none', edgecolor=(0, 0.8, 0.8), 
            linewidth=3, alpha=0.5)

    # what would you like the difference map to be?
    # center regions in 2d, covariance matrix
    # scaling factor for the probability at that point as the mean of another draw, with the mean centered at that number, sigma being very small
    regions = [{'mu': np.array([-1, -0.25]), 
                'cov':this_entropy_level*np.array([[0.1, -0.0], [-0, 1]])},
               {'mu': np.array([1, -0.25]), 
                'cov':this_entropy_level*np.array([[0.1, -0.0], [-0, 1]])},
               {'mu': np.array([-1, -0.75]), 
                'cov':this_entropy_level*np.array([[1, -0.0], [-0, 0.1]])},
               {'mu': np.array([1, -0.75]), 
                'cov':this_entropy_level*np.array([[1, -0.0], [-0, 0.1]])},
                ]

    x_values = np.array([np.linspace(-1,1,100).tolist()]*100)
    y_values = np.array([np.linspace(-1,1,100).tolist()]*100).transpose()
    xy_values = np.dstack([x_values, y_values])
    xy_values_flatten = xy_values.reshape(-1,2)

    probs = []
    weights = []
    for region in regions:
        probs.append ( multivariate_normal.pdf(xy_values_flatten, mean=region['mu'], cov=region['cov']))
        weights.append(1)

    # adding random noise 
    for i in range(6):
        if i%2 == 0:
            fake_mu = np.random.multivariate_normal(np.array([-1, -0.25]), 0.2*np.identity(2)) 
        else:
            fake_mu =np.random.multivariate_normal(np.array([1, -0.25]), 0.2*np.identity(2))  
        fake_cov = 0.08* np.identity(2) 
        print(fake_mu)
        print(fake_cov)
        probs.append ( multivariate_normal.pdf(xy_values_flatten , 
            mean = fake_mu,
            cov = fake_cov))
        weights.append(0.5);
    print(len(probs))
    weights = weights/np.sum(weights)
    prob = np.sum(np.array(weights)[:, None]* np.vstack(probs), 0)

    x = xy_values_flatten[:, 0]
    y = xy_values_flatten[:, 1]
    x_circle = x*np.sqrt(np.maximum(0.0000001, 1 - y**2/2))
    y_circle = y*np.sqrt(np.maximum(0.0000001, 1 - x**2/2))


    intensity = np.random.normal(prob, 0.05) 
    ax.scatter(x_circle, y_circle, c=intensity, cmap=cmap)
    ax.set_title('{} ms'.format(timestep*50))
    ax.axis('off') 

plt.savefig('head_scan.jpg')
