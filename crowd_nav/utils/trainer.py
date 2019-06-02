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
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

colors = ['r','g','b','y','k','c']

class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs, with_validation = False):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self.data_loader))

            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss

class TrajTrainer(Trainer):

    def __init__(self, model, train_memory, val_memory, device, batch_size, output_dir,logger):
        super().__init__(model, None, device, batch_size)
        self.train_memory = train_memory
        self.val_memory = val_memory
        self.train_data_loader = None
        self.val_data_loader = None
        self.output_dir = output_dir
        self.logger = logger
        self.fig, self.ax = plt.subplots(1,1,figsize=(7, 7))


    def update_plot(self, iteration):
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("L2 Loss")
        self.ax.set_title('IL training')
        self.ax.set_yscale('log')
        self.ax.plot(np.arange(iteration),self.val_loss,color=colors[0],label = 'Validation Loss')
        self.ax.plot(np.arange(iteration),self.train_loss,color=colors[1],label = 'Training Loss')
        self.ax.legend()
        plt.savefig(os.path.join(self.output_dir,'il_plot.jpg'))
        np.save(os.path.join(self.output_dir,'val_loss'),np.asarray(self.val_loss))
        np.save(os.path.join(self.output_dir,'train_loss'),np.asarray(self.train_loss))
        plt.cla()


    def optimize_epoch(self, num_epochs, robot, with_validation = False):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.train_data_loader is None:
            self.train_data_loader = DataLoader(self.train_memory, self.batch_size, shuffle=True)
        if self.val_data_loader is None:
            self.val_data_loader = DataLoader(self.val_memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        self.val_loss = []
        self.train_loss = []
        for epoch in range(num_epochs):
            train_loss = 0
            iteration = 0
            #distances = []

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
                #dist = np.linalg.norm(self_info[:,:2]-self_info[:,5:7],axis=1)
                #distances.append(dist)
                outputs = self.model((trajs, rel_trajs, self_info), to_convert = False)
                #if epoch == num_epochs-1:
                #    print('Output ',outputs)
                #    print('values', values)

                #print(self.model(self.model.last_input, to_convert = False, policy_learning=policy_learning))
                loss = self.criterion(outputs, values) # Convert action to 2D

                loss.backward()

                self.optimizer.step()

                #raise NotImplementedError

                train_loss += loss.data.item()

            val_loss = 0
            j = 0
            # distances = np.array(distances)
            # print(distances)
            # print(np.histogram(distances))
            # print(np.min(distances))
            # print(np.max(distances))
            # print(len(distances[distances<0.3]))
            # raise NotImplementedError
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

                outputs = self.model((trajs, rel_trajs, self_info), to_convert = False)
                batch_loss = self.criterion(outputs, values) # Convert action to 2D
                val_loss += batch_loss.data.item()
            self.logger.debug('Average loss on validation set on epoch %02d: %.2E', epoch, val_loss/j)
            self.logger.debug('Average loss on test set on epoch %02d: %.2E', epoch, train_loss/iteration)
            self.val_loss.append(val_loss/j)
            self.train_loss.append(train_loss/iteration)


            train_loss = 0

            # print(values.shape)
            # print('Value:')
            # print(values[0,:])
            # print('Output:')
            # print(outputs[0,:])
            # print('Loss:')
            # print(loss.data.item())
            # input()
            average_epoch_loss = train_loss / iteration
            self.update_plot(epoch+1)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.train_memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self.data_loader))

            obs_traj = inputs[0].squeeze(1)
            obs_traj_rel = inputs[1].squeeze(1)
            self_state = inputs[2].squeeze(1)
            obs_traj = Variable(obs_traj)
            obs_traj_rel = Variable(obs_traj_rel)
            self_state = Variable(self_state)
            values = Variable(values)

            self.optimizer.zero_grad()
            outputs = self.model((obs_traj, obs_traj_rel, self_state), to_convert = False)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        self.logger.debug('Average loss : %.2E', average_loss)

        return average_loss