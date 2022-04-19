'''
BloodViscoVelocityPINN: The network is inspired from the Raissi's 2019 PINN paper. 

PINN Paper: https://www.sciencedirect.com/science/article/pii/S0021999118307125
Official code: https://github.com/maziarraissi/PINNs
Authors: Maziar Raissi, Paris Perdikaris, and George Em Karniadakis

@author: Hemanthika A
''' 
# import libraries
from audioop import bias
from dataclasses import replace
from pickletools import optimize
from turtle import color
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import grad
import numpy as np
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Neural Network Initialization
class NN(nn.Module):
    '''
    This class is used to initialize the neural network used in the project
    '''
    def __init__(self, layers):
        super(NN, self).__init__()
        # initializations
        self.activation = torch.nn.Tanh
        layer_list = []
        for i in range(len(layers) - 2): 
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1], bias=True)))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (len(layers) - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)

        return out

# Main Class
class BloodViscoVelocityPINN():
    def __init__(self, train_labelled, train_complete, layers):
        '''
        All initializations required before training are mentioned here
        '''
        self.x = torch.tensor(train_complete[:, 0:1], requires_grad=True).float()
        self.t = torch.tensor(train_complete[:, 1:2], requires_grad=True).float()
        self.p = torch.tensor(train_labelled[:,2:3]).float() # labelled pressure
        self.u = torch.tensor(train_labelled[:,3:4]).float() # labelled velocity
        self.NL = len(train_labelled)
        l1 = float(np.random.randint(low=1000, high=1100, size=(1,))[0])
        l2 = float(np.random.randint(low=3.5, high=5.5, size=(1,))[0])
        print(f"Lambda1: {l1}, lambda2={l2}")
        self.lambda1 = torch.tensor([l1], requires_grad=True).float()
        self.lambda2 = torch.tensor([l2], requires_grad=True).float()
        self.lambda1 = nn.Parameter(self.lambda1) # can add to model params
        self.lambda2 = nn.Parameter(self.lambda2)
        self.layers = layers
        self.iter = 0
        # model initialization
        self.nn = NN(self.layers)
        self.nn = self.init_weights(self.nn)
        self.nn.register_parameter('lambda_1', self.lambda1)
        self.nn.register_parameter('lambda_2', self.lambda2)
        # optimizer
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.nn.parameters(), 
            lr=1.0, 
            max_iter=1500, 
            max_eval=1500, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.nn.parameters(),lr=1)

    def init_weights(self, network):
        '''
        initialize weights as xavier weights
        '''
        if isinstance(network, nn.Linear):
            nn.init.xavier_uniform(network.weight)
            network.bias.data.fill_(0.01)

        return network

    def u_net(self, network):
        '''
        Velocity network u(x,t)
        '''
        u_net = network[:,0:1]

        return u_net

    def p_net(self, network):
        '''
        Pressure network p(x,t)
        '''
        p_net = network[:,1:2]

        return p_net

    def PINN(self, x, t):
        '''
        Complete network which represents homogenous form of PDE
        '''
        X = torch.cat([x, t], dim=1)
        net = self.nn(X)
        u_net = self.u_net(net)
        p_net = self.p_net(net)
        u_net_x = grad(u_net,x,grad_outputs=torch.ones_like(u_net),retain_graph=True,create_graph=True)[0] # du/dx
        u_net_xx = grad(u_net_x,x,grad_outputs=torch.ones_like(u_net_x),retain_graph=True,create_graph=True)[0] # du/dxdx
        u_net_t = grad(u_net,t,grad_outputs=torch.ones_like(u_net),retain_graph=True,create_graph=True)[0] # du/dt
        p_net_x = grad(p_net,x,grad_outputs=torch.ones_like(p_net),retain_graph=True,create_graph=True)[0] # dp/dx
        f = u_net_t + (u_net * self.lambda1 * u_net_x) + p_net_x - (self.lambda2 * u_net_xx)

        return u_net, p_net, f

    def loss(self, p_pred, f):
        '''
        Computes custom loss of the network which includes physics laws
        ''' 
        # import pdb;pdb.set_trace()
        total_loss = torch.sum((self.p - p_pred[:self.NL,:]) ** 2) + torch.sum(f ** 2)

        return total_loss

    def train(self, niter):
        '''
        Train function.
        '''
        self.nn.train() # set training phase
        for epoch in range(niter):
            u_pred, p_pred, f = self.PINN(self.x, self.t) # predictions
            print(min(u_pred), max(u_pred))
            loss = self.loss(p_pred,f) # model loss
            # update parameters
            self.optimizer_Adam.zero_grad() # use adam optimizer for if data is noisy
            loss.backward()
            self.optimizer_Adam.step()
            loss_u = torch.sum((self.u - u_pred[:self.NL,:]) ** 2) # velocity loss
            if epoch%1==0:
                print(f"Adam: Epoch: {epoch}/{niter}\t Train Loss: {loss}\t U_pred Loss: {loss_u}\t l1: {self.lambda1.item()}\t l2: {self.lambda2.item()}")
        def closure():
            self.optimizer_LBFGS.zero_grad()
            u_pred, p_pred, f = self.PINN(self.x, self.t) # predictions
            loss = self.loss(p_pred,f)
            loss.backward()
            self.iter += 1 
            loss_u = torch.sum((self.u - u_pred[:self.NL,:]) ** 2) # velocity loss
            if self.iter%100==0:
                print(f"LBFGS: Epoch: {self.iter}\t Train Loss: {loss}\t U_pred Loss: {loss_u}\t l1: {self.lambda1.item()}\t l2: {self.lambda2.item()}")
            return loss
        self.optimizer_LBFGS.step(closure)

    def predict(self,X):
        '''
        Predict function for test data
        '''
        x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        t = torch.tensor(X[:, 1:2], requires_grad=True).float()
        self.nn.eval()
        u_pred, p_pred, f = self.PINN(x, t)

        return u_pred.detach().numpy(), p_pred.detach().numpy()


# Main code
if __name__=="__main__":
    '''
    The main function. code execution will start from here
    '''
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2] # define layers of the neural network
    # data loading
    x = np.load("./Data/x.npy") # x coordinates
    t = np.load("./Data/t.npy") # time values
    p = np.load("./Data/p.npy") # pressure values
    u = np.load("./Data/u.npy") # velocity values
    xc = np.load("./Data/xc.npy") # x coordinates of collocation points
    tc = np.load("./Data/tc.npy") # time values of collocation points

    # here idea is to create two training sets. 1 for labelled data 2 for complete data which includes collocation points also
    #### LABELLED DATA ####
    # initial condition t=0
    xi = x
    ti = np.full((len(t),1),0)
    pi = np.full((len(t),1),p[0])
    ui = np.full((len(t),1),u[0])
    # boundary condition x=0
    xb = np.full((len(t),1),0)
    tb = t
    pb = p
    ub = u

    # labelled data
    train_labelled_1 = np.concatenate([xi,ti,pi,ui], axis=1)
    train_labelled_2 = np.concatenate([xb,tb,pb,ub], axis=1)
    train_labelled = np.concatenate([train_labelled_1,train_labelled_2], axis=0) # size = (2*len(t))*4
    np.random.shuffle(train_labelled) # shuffle data
    # train_labelled = train_labelled / train_labelled.max(axis=0) # normalize

    # complete data: labelled + collocation points
    train_complete_1 = train_labelled[:,0:2]
    train_complete_2 = np.concatenate([xc,tc], axis=1)
    train_complete = np.concatenate([train_complete_1,train_complete_2], axis=0) # size = N*2

    N_labelled = len(train_labelled) # labelled data size
    N_collocation = len(xc) # collocation data size

    #### MODELLING ####
    model = BloodViscoVelocityPINN(train_labelled, train_complete, layers)
    model.train(10)
    u_pred, p_pred = model.predict(np.concatenate([xb,tb], axis=1))
    # u_pred = u_pred * train_labelled.max(axis=0)[-1]

    print(train_labelled.max(axis=0)[-1])

    # Ploting
    fig = plt.figure(figsize=(14, 10))
    plt.plot(tb,u_pred, linestyle = '--', color='red')
    plt.plot(tb,u, linestyle = '-', color='blue')

    plt.savefig('plot.jpeg')











