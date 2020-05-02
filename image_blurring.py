import matplotlib.pyplot as plt
import numpy as np
import cv2
max_variance = 300 
variances = np.linspace(0,max_variance,10) # 10 epochs

fig = plt.figure(figsize = (10, 6))
gs = fig.add_gridspec(5, 10)

row_number = 0
for fruit in ['apples.jpeg', 'banana.jpg', 'blueberries.jpeg', 'mango.jpg', 'pear.jpg']:
    img = cv2.imread(fruit)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    column_number = 0

    for var in variances:
        noise = np.random.normal(0, var, img.shape)
        ax2 = fig.add_subplot(gs[row_number, column_number]) 
        ax2.imshow(np.minimum(np.maximum((img+noise).astype(int), 0), 255))
        ax2.set_title('Var: {}'.format(int(var)))
        ax2.axis('off')
        column_number+=1
    row_number += 1 

plt.savefig('images_noise.jpg')

