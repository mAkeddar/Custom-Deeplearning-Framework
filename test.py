#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:17:10 2020
@course: 559DL
@file  : MP N#2
@author: 261344, 261864, 260496
"""

import math #standard Math library
import torch #we can't just import torch.empty because otherwise we can't do the following:
from torch import empty as torch_empty

#The thing that rules the project
torch.set_grad_enabled(False)


######SCRIPT PARAMETER:
PLOT_GRAPHS_PERF = True #PLOT GRAPHS?
COMPUTE_TIME_TRAIN = False #COMPUTE TRAINING TIME?

if PLOT_GRAPHS_PERF:
    from matplotlib import pyplot as plt
if COMPUTE_TIME_TRAIN:
    import time

#NETWORK WIDTHS
hiddenLayerWidth = 25
outputDim = 2 

#GLOBAL PARAMS
fillValue = 1e-6 #default value for grad
zeta = 0.9 #labels scale down
lr = 0.03 #Learning rate

###########################
######### Module & children classes
class Module(object):
    #This is the Base Class, each module contains these functions
    def __init__(self):
        super(Module, self).__init__()
        self.input_ = None

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return []

    def info(self):
        raise NotImplementedError 
    
class LinearLayer(Module):
    
    # This class creates a Linear node
    #It has 3 Arguments:
       #in_dim : number of nodes that are incident to the layer
       # out_dim : number of nodes that the network outputs
       #Biais : allows bias
    #It is defined by f(x) = w*x+b
    def __init__(self, in_dim, out_dim, biais=True):
        super(LinearLayer, self).__init__()

        std = math.sqrt(2 / (in_dim + out_dim))
        self.w = torch.empty(out_dim, in_dim).normal_(0, std)
        self.b = torch.empty(1,out_dim).normal_(0, std)
        self.dw = torch.empty(self.w.shape).fill_(fillValue)
        self.db = torch.empty(self.b.shape).fill_(fillValue)
        #change the input to optimize the output
        self.useBiais = biais
        if (not self.useBiais):
            self.b = self.b.zero_()
    
    def forward(self, x):
        #This function implements the forward pass for the linear module.
        #computes y as f(x)= <w,x> + b
        #Also saves X and Y as self.input and self.output for backward pass
        self.input_ = x.clone()
        y = self.input_.mm(self.w.t())
        if self.useBiais:
            y = y + self.b
        self.output = y
        return y
    
    def backward(self, grad_output):
        #This function implements the Backward pass for the linear module
        #
        self.dw += self.input_.t().mm(grad_output).t().div(self.input_.shape[0])
        if self.useBiais:
            self.db += grad_output.sum(0).div(self.input_.shape[0]) #dl_ds
        self.input_=None
        return grad_output.mm(self.w)
    
    def param(self):
        return [(self.w,self.dw), (self.b,self.db)]  

    def info(self,st=''):
        #Displays the dimension of the layer when we print the network
        print(st+"Linear Layer, in: " + str(self.w.shape[1]) +\
              ", out: " + str(self.w.shape[0]) + ", biais: " + str(self.useBiais))
        
class nnSequential(Module):
    #This class takes the model as argument and creates a multilayer perceptron with different layers composed of linear layer with non linear activation functions
    def __init__(self, *model):
        super(nnSequential, self).__init__()

        if not (type(model) == tuple):
            raise TypeError
        
        self.net =  model
        
    def forward(self, x):
        #calls the appropriate forward pass function from the correct module according to the info of the *model
        self.input_ = x.clone()
        out = self.input_
        for net in self.net:
            out = net.forward(out)        
        return out
    
    def backward(self, dloss):
        #calls the appropriate backward pass function from the correct module according to the info of the *model
        grad = dloss
        for layer in reversed(self.net):
            grad = layer.backward(grad)

        self.input_ = None
        return grad
        
   

    def info(self,st='| '):
        #This function prints our network
        print(st[:len(st)-1]+"x Sequential:")
        st = ' ' + st #+ '| '

        for layer in self.net:
            layer.info(st)
        print(st[:len(st)-2]+"V")
        
    def param(self):
        param_list = []
        for layer in self.net:
            param_list.extend(layer.param())

        return param_list
        
class ReLU(Module):
    #This class implements the Rectified Linear Unit function, it's a non-linear activation function 
    def __init__(self):
        self.input_ = None

    def forward(self, x):
        #Computes ReLU(X) and returns it        
        self.input_ = x.clone()
        zeros = torch_empty(self.input_.size()).zero_()
        return self.input_.max(zeros)

    def backward(self, grad_output):
        #Computes the gradient of ReLU and returns it
        grad_input = (self.input_ > 0).float().mul(grad_output)
        self.input_ = None
        return grad_input
        
    def info(self,st=''):
        #Displays the name of the activation function when we print the network
        print(st+"Rectified Linear Unit")
    
class tanh(Module):
    #This class implements the Hyperbolic Tangent, it's a non-linear activation function 
    def __init__(self):
        self.input_ = None
        
    def forward(self,x):
        #Computes Tanh(X) and returns it      
        self.input_ = x.clone()
        return self.input_.tanh()
    
    def backward(self,grad_output):
        #Computes the gradient of tanh and returns it
        grad_input = 1-self.input_.tanh().pow(2).mul(grad_output)
        self.input_ = None
        return grad_input
    
    def info(self,st=''):
        #Displays the name of the activation function when we print the network
        print(st+"Hyperbolic Tangent")
    
    
class sigmoid(Module):
    #This class implements the sigmoid func 
    def __init__(self):
        self.input_ = None
        
    def forward(self,x):
        #Computes Tanh(X) and returns it      
        self.input_ = x.clone()
        return self.input_.sigmoid()
    
    def backward(self,grad_output):
        #Computes the gradient of tanh and returns it
        grad_input = self.input_.sigmoid()*(1-self.input_.sigmoid()).mul(grad_output)
        self.input_ = None
        return grad_input
    
    def info(self,st=''):
        #Displays the name of the activation function when we print the network
        print(st+"Sigmoid")

class MSEloss(Module):
    #This class implements the MSE loss 
    def __init__(self):
        self.input_ = None
        self.target = None
        
    def forward(self, input_, target):
        #computes the l2-norm loss betweem the input and the target
        self.input_ = input_
        self.target = target
        delta = input_ - target
        return delta.pow(2).mean()

    def backward(self):
        #computes the gradient of the input 
        delta = self.input_ - self.target
        n = delta.size()[1]
        grad_input = 2*delta.div(n)
        
        self.input_ = None
        self.target = None
        return grad_input
    def info(self):
        print("Mean Square Error Loss")

###########################
######### Optimizer class& children classes
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param[1].zero_()

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, clip=None):
        super(SGD, self).__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.clip = clip

        if (self.momentum != 0):
            self.Vt_prev_buffer = []
            for _, p_grad in self.params:
                self.Vt_prev_buffer.append(p_grad)#.clone())

    def step(self,model):
        for i, (p, p_grad) in enumerate(self.params):
            # FROM PYTORCH https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
            if (self.momentum != 0 and self.momentum is not None):
                Vt = self.momentum*self.Vt_prev_buffer[i] +  p_grad
                self.Vt_prev_buffer[i] = Vt

            if self.clip is not None:
                Vt[Vt>self.clip] = self.clip
                Vt[Vt<-self.clip] = -self.clip

            # update parameter
            p.add_(-self.lr*Vt)

###########################
######### Usual functions
def train(model,train_input,train_target, nb_epochs,optimizer,lossfun, mini_batch_size, te_dat=None, te_tar=None):
    
    err_tr=[]
    err_te=[]
    
    for e in range(0, nb_epochs):
        sum_loss=0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = lossfun(output, train_target.narrow(0, b,mini_batch_size))
            optimizer.zero_grad()
            model.backward(lossfun.backward())
            optimizer.step(model)
            sum_loss += loss.item()
            
        if PLOT_GRAPHS_PERF:
            err_tr.append(compute_nb_errors(model,train_input,train_target,mini_batch_size))
            err_te.append(compute_nb_errors(model,te_dat,te_tar,mini_batch_size))
        print('e', e, '- loss:', sum_loss)
        
    return err_tr,err_te

def compute_nb_errors(model,data_input, data_target,mini_batch_size, pltCross=False):
        
    nb_data_errors =  0
    cl_tr = torch.argmax(data_target,1)
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        predicted_classes = torch.argmax(output, 1)
        for k in range(mini_batch_size):
            if cl_tr[b + k] != predicted_classes[k]:
                if pltCross and PLOT_GRAPHS_PERF:
                    plt.scatter(data_input[b+k,0],data_input[b+k,1],marker="x", c='r')
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

###########################
######### Functions to generate & deal with dataset
def target_to_onehot(target):
    #This function applies onehot encoding to transform the matrices to vectors
    res = torch.empty(target.size(0), 2).zero_()
    res.scatter_(1, target.view(-1, 1), 1.0)
    return res

def generate_disc_set(nb, balance=True):
    #This function generates the data with its associated labels
    data=torch.rand(nb,2)
    labels = torch.norm(data-0.5,p=2,dim=1) < math.sqrt(1/(2*math.pi))
    
    # classes are not even? recursive call !
    if (balance and torch.abs(torch.sum(labels == True)-torch.sum(labels == False)) >= math.log(nb)/math.log(10)):
        data,labels = generate_disc_set(nb)
    
    std = torch.std(data,dim=0)
    mean = torch.mean(data,dim=0)
    data = torch.div(data-mean, std)
    return data,labels.type(torch.LongTensor)

#############################################################
###########################
######### Main program
N_tr = 1000
N_te = 1000
nb_epochs = 100
mini_batch_size = 100

momentum = 0.9
clip = 5

data_tr, labels_tr = generate_disc_set(N_tr)
data_te, labels_te = generate_disc_set(N_te, False)

labels_tr = target_to_onehot(labels_tr)*zeta
labels_te = target_to_onehot(labels_te)*zeta


#Just to demonstrate nested sequential layers, supported by framework
modelp1 = nnSequential(LinearLayer(data_tr.shape[1],hiddenLayerWidth), ReLU(),\
                     LinearLayer(hiddenLayerWidth,hiddenLayerWidth, False), ReLU())
modelp2 = nnSequential(LinearLayer(hiddenLayerWidth,hiddenLayerWidth, False), ReLU(),\
                     LinearLayer(hiddenLayerWidth,outputDim), ReLU())
model = nnSequential(modelp1, modelp2)

optimizer = SGD(model.param(), lr, momentum, clip)
loss = MSEloss()

if COMPUTE_TIME_TRAIN:
    tim1 = time.perf_counter()
    
err_tr,err_te = train(model,data_tr,labels_tr,nb_epochs,optimizer,loss,\
                      mini_batch_size,data_te,labels_te)

if COMPUTE_TIME_TRAIN:
    tim2 = time.perf_counter()
    print('Learning time: {:e} [s]'.format((tim2 - tim1)))
if PLOT_GRAPHS_PERF:
    plt.title("Train set")
    plt.scatter(data_tr[labels_tr[:,0]==zeta,0],data_tr[labels_tr[:,0]==zeta,1], c='black')
    plt.scatter(data_tr[labels_tr[:,0]==0,0],data_tr[labels_tr[:,0]==0,1], c='0.75')
    plt.axis('equal')

error = compute_nb_errors(model,data_tr,labels_tr,mini_batch_size,pltCross=True)

if PLOT_GRAPHS_PERF:
    plt.show()
    plt.title("Test set")
    plt.scatter(data_te[labels_te[:,0]==zeta,0],data_te[labels_te[:,0]==zeta,1], c='black')
    plt.scatter(data_te[labels_te[:,0]==0,0],data_te[labels_te[:,0]==0,1], c='0.75')
    plt.axis('equal')

terror = compute_nb_errors(model,data_te,labels_te,mini_batch_size,pltCross=True)

if PLOT_GRAPHS_PERF:
    plt.show()
    plt.title("Error(epoch)")
    plt.plot(range(0,nb_epochs),err_tr,label='Train error')
    plt.plot(range(0,nb_epochs),err_te,label='Test error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylabel('Number of errors')
    plt.show()

print('Training error: ' + str(error), end='')
print(' - Testing error: ' + str(terror))