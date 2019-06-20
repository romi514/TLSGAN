import logging
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
#matplotlib.use('Agg') # For plot generation, comment out for simulator rendering
import matplotlib.pyplot as plt 

colors = ['r','g','b','y','k','c']

class Trainer(object):

    def __init__(self, model, train_memory, val_memory, device, batch_size, output_dir,logger):
        self.train_memory = train_memory
        self.val_memory = val_memory
        self.train_data_loader = None
        self.val_data_loader = None
        self.output_dir = output_dir
        self.logger = logger
        self.fig, self.ax = plt.subplots(1,1,figsize=(7, 7))

        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)


    def update_plot(self, iteration):
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("L2 Loss")
        self.ax.set_title('IL training')
        self.ax.set_yscale('log')
        self.ax.plot(np.arange(iteration+1),self.val_loss,color=colors[0],label = 'Validation Loss')
        self.ax.plot(np.arange(iteration+1),self.train_loss,color=colors[1],label = 'Training Loss')
        self.ax.legend()
        plt.savefig(os.path.join(self.output_dir,'il_plot.jpg'))
        np.save(os.path.join(self.output_dir,'val_loss'),np.asarray(self.val_loss))
        np.save(os.path.join(self.output_dir,'train_loss'),np.asarray(self.train_loss))
        plt.cla()


    def optimize_epoch_il(self, num_epochs, robot, with_validation = False):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.train_data_loader is None:
            self.train_data_loader = DataLoader(self.train_memory, self.batch_size, shuffle=True)
        if self.val_data_loader is None:
            self.val_data_loader = DataLoader(self.val_memory, 1, shuffle=True)
        average_epoch_loss = 0
        self.val_loss = []
        self.train_loss = []
        for epoch in range(num_epochs):
            train_loss = 0
            iteration = 0

            # Train on training data
            for data in self.train_data_loader:
                iteration+=1
                inputs, values = data
                trajs = inputs[0].squeeze(1)
                rel_trajs = inputs[1].squeeze(1)
                self_info = inputs[2].squeeze(1)

                trajs = Variable(trajs)
                rel_trajs = Variable(rel_trajs)
                self_info = Variable(self_info)

                values = Variable(values)

                self.optimizer.zero_grad()

                outputs = self.model((trajs, rel_trajs, self_info))

                loss = self.criterion(outputs, values)

                loss.backward()

                self.optimizer.step()


                train_loss += loss.data.item()

            # Get validation loss
            val_loss = 0
            j = 0
            for val_data in self.val_data_loader:
                j += 1
                inputs, values = val_data
                trajs = inputs[0].squeeze(1)
                rel_trajs = inputs[1].squeeze(1)
                self_info = inputs[2].squeeze(1)

                trajs = Variable(trajs)
                rel_trajs = Variable(rel_trajs)
                self_info = Variable(self_info)

                values = Variable(values)

                outputs = self.model((trajs, rel_trajs, self_info))
                batch_loss = self.criterion(outputs, values) # Convert action to 2D
                val_loss += batch_loss.data.item()

            # Log results
            self.logger.debug('Average loss on validation set on epoch %02d: %.2E', epoch, val_loss/j)
            self.logger.debug('Average loss on test set on epoch %02d: %.2E', epoch, train_loss/iteration)
            self.val_loss.append(val_loss/j)
            self.train_loss.append(train_loss/iteration)

            self.update_plot(epoch)

    def optimize_batch_rl(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.train_loader is None:
            self.train_loader = DataLoader(self.train_memory, self.batch_size, shuffle=True)

        losses = 0
        for _ in range(num_batches):

            raise NotImplementedError

        average_loss = losses / num_batches
        self.logger.debug('Average loss : %.2E', average_loss)
