import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import cv2
import pandas as pd

max_variance = 300
def noise_generator(var):
    return 300*var/max_variance + 100

heights = [20, 5]
fig = plt.figure(figsize=(25, 10))
gs = fig.add_gridspec(2, 10, height_ratios = heights)

variances = np.linspace(0,max_variance,10) # 10 epochs

y_limit = 1000
# y_known = np.array([np.exp(x*5e-3) * 180 for x in variances] )
y_known = np.array([ 17500/(320-x)+ 125 for x in variances])
delta = np.vstack([np.random.normal(0, noise_generator(element), 14) for element in variances]).transpose() # 14 monkeys
y_observed = y_known + delta

variances_mesh , monkey_mesh= np.meshgrid (variances, np.arange(14)) 
data_dict = { 'subject': monkey_mesh.flatten(),
                'variance': variances_mesh.flatten(), 
                'reaction':y_observed.flatten()}
dataset = pd.DataFrame(data=data_dict)
ax1 = fig.add_subplot(gs[0,:])
sns.lineplot(x='variance', y='reaction', data=dataset, ax=ax1)
# plt.plot( variances, y_observed.transpose() )
ax1.set_ylim(bottom= 0, top=1000)

# generating images
img = cv2.imread('banana.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

counter = 0
for var in variances:
    print(var)
    noise = np.random.normal(0, var, img.shape)
    ax2 = fig.add_subplot(gs[1, counter]) 
    ax2.imshow(np.minimum(np.maximum((img+noise).astype(int), 0), 255))
    ax2.set_title('Variance: {}'.format(int(var)))
    ax2.axis('off')
    counter += 1 

plt.savefig('reaction_to_noise.jpg')

