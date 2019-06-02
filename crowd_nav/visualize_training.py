import matplotlib.pyplot as plt
import numpy as np

#rl_outputfiles = ["with_pool_output.log",'no_pool_output.log','sarl_output.log']
files = ['data/output11/output.log','data/output12/output.log']
# il_outputfiles = ["data/Pool_PreTrain_RL/output2.log",'data/NoPool_PreTrain_IL/output.log','data/NoPool_Rand_IL/output.log','data/SARL_IL/output.log','data/NoPool_PreTrain_IL_obs4/output.log']
# labels = ['Frozen with Pooling','Frozen without Pooling','Random init','Sarl','Frozen without Pooling - 4 len obs']
labels = ['Simple Validation Loss','Simple Training Loss','Pretrained Validation Loss','Pretrained Training Loss','Pretrained Validation Loss - 0.001 LR','Pretrained Training Loss - 0.001 LR']
colors = ['r','g','b','y','k','c']

fig, ax = plt.subplots(1,1,figsize=(7, 7))
plt.subplots_adjust(hspace = 0.5)
ax.set_xlabel("Iterations")
ax.set_ylabel("L2 Loss")
ax.set_xticks([1000,2000,2800,4000,5200,6400], minor=False)
ax.xaxis.grid(True, which='minor')
ax.set_title('IL Policy training of Random SGAN')
ax.set_yscale('log')


for i,name in enumerate(files):
	factor = 10
	with open(name, "r") as f:
		content = f.readlines()

	val_losses = [line for line in content if "Average loss on validation set" in line]
	val_losses = np.asarray([float(line[line.find("Average loss on validation set")+55:]) for line in val_losses])
	#if len(il_losses) != 0:
	#	il_losses = np.concatenate((il_losses, np.random.normal(il_losses[-1], 1.50E-06, 50-len(il_losses))))
	t = [factor*i for i in range(0, len(val_losses))]
	ax.plot(t[:],val_losses[:],color=colors[2*i], label = labels[2*i])

	train_losses = [line for line in content if "Average loss on test set" in line]
	#print(float(rl_losses[0][rl_losses[0].find("Average loss on test set")+50:]))
	train_losses = [float(line[line.find("Average loss on test set")+49:]) for line in train_losses]

	t = [factor*i for i in range(0, len(train_losses))]
	ax.plot(t[:],train_losses[:],color=colors[2*i+1],label = labels[2*i+1])
	ax.legend()

plt.show()
