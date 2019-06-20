import matplotlib.pyplot as plt
import numpy as np
import os

files = ['data/output13','data/output14','data/output13']
labels = ['Frozen Validation Loss','Frozen Training Loss','Pretrained Validation Loss','Pretrained Training Loss','Random Validation Loss','Random Training Loss']
colors = ['r','g','b']

fig, ax = plt.subplots(1,1,figsize=(7, 7))
ax.set_xlabel("Iterations")
ax.set_ylabel("L2 Loss")
ax.set_title('IL Policy training of different learning methods')
ax.set_yscale('log')

for i,name in enumerate(files):
	
	val_loss = np.load(os.path.join(name,'val_loss.npy'))
	train_loss = np.load(os.path.join(name,'train_loss.npy'))

	t = np.arange(len(val_loss))

	ax.plot(t,val_loss,'--'+colors[i], label = labels[2*i])
	ax.plot(t,train_loss,colors[i],label = labels[2*i+1])
	ax.legend()

plt.show()