'''
BloodViscoVelocityPINN: The network is inspired from the Raissi's 2019 PINN paper. 

PINN Paper: https://www.sciencedirect.com/science/article/pii/S0021999118307125
Official code: https://github.com/maziarraissi/PINNs
Authors: Maziar Raissi, Paris Perdikaris, and George Em Karniadakis

@author: Hemanthika A
''' 

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import figure

np.random.seed(1234)
tf.set_random_seed(1234)

class BloodViscoVelocityPINN:
    # Initialize the class
    def __init__(self, xi, ti, xb, xt, x_f, t_f, pi, pb, ui, ub, layers):  
        
        Xi = np.concatenate([xi, ti], 1)
        Xb = np.concatenate([xb, tb], 1)
        Xf = np.concatenate([x_f, t_f], 1)
        
        self.lb = Xi.min(0)
        self.ub = Xb.max(0)
                
        self.Xi = Xi
        self.Xb = Xb
        self.Xf = Xf
        
        self.xi = Xi[:,0:1]
        self.ti = Xi[:,1:2]
        self.xb = Xb[:,0:1]
        self.tb = Xb[:,1:2]
        self.xf = Xf[:,0:1]
        self.tf = Xf[:,1:2]
        self.pi = pi
        self.pb = pb
        self.ui = ui
        self.ub = ub
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize parameters
        # l1val = tf.random.uniform(shape=[1], minval=0, maxval=0.5) 
        # self.lambda_1 = tf.Variable(l1val, dtype=tf.float32)
        # l2val = tf.random.uniform(shape=[1], minval=0, maxval=0.000015)
        # self.lambda_2 = tf.Variable(l2val, dtype=tf.float32)

        self.lambda_1 = tf.Variable([0.01], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.01], dtype=tf.float32)

        self.A = 2.293820e-04
        self.U = 1e+1

        # Charactiristic variables for non-dimensionalization
        self.L = np.sqrt(self.A)
        self.T = self.L/self.U
        self.p0 = self.lambda_1*self.U**2

        # Non-dimensionalize
        self.A0 = self.L**2    

        #  = X_measurement_aorta1/self.L
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_i_tf = tf.placeholder(tf.float32, shape=[None, self.xi.shape[1]])
        self.t_i_tf = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.tf.shape[1]])
        self.p_i_tf = tf.placeholder(tf.float32, shape=[None, self.pi.shape[1]])
        self.p_b_tf = tf.placeholder(tf.float32, shape=[None, self.pb.shape[1]])
        self.u_i_tf = tf.placeholder(tf.float32, shape=[None, self.ui.shape[1]])
        self.u_b_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        
        self.u_i_pred = self.net(self.x_i_tf, self.t_i_tf)
        self.u_b_pred = self.net(self.x_b_tf, self.t_b_tf)
        self.p_i_pred = self.net(self.x_i_tf, self.t_i_tf)
        self.p_b_pred = self.net(self.x_b_tf, self.t_b_tf)
        self.a_i_pred = self.net(self.x_i_tf, self.t_i_tf)
        self.a_b_pred = self.net(self.x_b_tf, self.t_b_tf)
        self.f1, self.f2, self.f3 = self.net_f(self.x_f_tf, self.t_f_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.p_i_tf - self.p_i_pred)) + \
                    tf.reduce_mean(tf.square(self.p_b_tf - self.p_b_pred)) + \
                    tf.reduce_mean(tf.square(self.f1)) + \
                    tf.reduce_mean(tf.square(self.f2)) + \
                    tf.reduce_mean(tf.square(self.f3))

        self.adam_loss_history = []
        self.adam_l1_history = []
        self.adam_l2_history = []
        self.lbfgs_loss_history = []
        self.lbfgs_l1_history = []
        self.lbfgs_l2_history = []
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 500000,
                                                                           'maxfun': 500000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def get_equilibrium_cross_sectional_area_aorta_1(self, x):
        x = self.L*(self.Xstd1*x + self.Xmean1)
        X1 = 0.
        X2 = 0.04964
        denom = X2-X1
        x1 = 2.293820e-04
        x2 = 2.636589e-04
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y

    def get_beta(self, x):
        x = self.L*(self.Xstd1*x + self.Xmean1)
        X1 = 0.
        X2 = 0.04964
        denom = X2-X1
        x1 = 2.472667e+06
        x2 = 2.151208e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net(self, x, t):
        u = self.neural_net(tf.concat([x,t], 1), self.weights, self.biases)

        return u
        
    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        
        u = self.net(x,t)
        p = self.net(x,t)
        a = self.net(x,t)

        a_t = tf.gradients(a, t)[0]
        au_x = tf.gradients(tf.multiply(a,u), t)[0]
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        
        p_x = tf.gradients(p, x)[0]

        f1 = a_t + au_x
        f2 = a
        
        return f_u
    
    def callback(self, loss, lambda_1, lambda_2):
        self.lbfgs_loss_history.append(loss)
        self.lbfgs_l1_history.append(lambda_1)
        self.lbfgs_l2_history.append(lambda_2)
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))
      
    def train(self, nIter): 

        tf_dict = {self.x_u_tf: self.xu, self.t_u_tf: self.tu, self.p_tf: self.pu,
                    self.x_f_tf: self.xf, self.t_f_tf: self.tf}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
                self.adam_loss_history.append(loss_value)
                self.adam_l1_history.append(lambda_1_value)
                self.adam_l2_history.append(lambda_2_value)

        # figure(figsize=(20, 5), dpi=80)

        plt.subplot(1,3,1)
        plt.plot(self.adam_loss_history)
        # x1,x2,y1,y2 = plt.axis()  
        # plt.axis((x1,x2,0,5000))
        plt.title("ADAM Loss History")
        # plt.savefig("./adam_loss_history.jpg")
        # plt.clf()
        plt.subplot(1,3,2)
        plt.plot(self.adam_l1_history)
        plt.title("ADAM Lambda1 History")
        # plt.savefig("./adam_l1_history.jpg")
        # plt.clf()
        plt.subplot(1,3,3)
        plt.plot(self.adam_l2_history)
        plt.title("ADAM Lambda2 History")
        
        plt.tight_layout()
        plt.savefig("./adam.jpg")
        plt.clf()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_1, self.lambda_2],
                                loss_callback = self.callback)
            
        figure(figsize=(20, 5), dpi=80)

        plt.subplot(1,3,1)
        plt.plot(self.lbfgs_loss_history)
        plt.title("LBFGS Loss History")
        # x1,x2,y1,y2 = plt.axis()  
        # plt.axis((x1,x2,0,5000))
        # plt.savefig("./lbfgs_loss_history.jpg")
        # plt.clf()
        plt.subplot(1,3,2)
        plt.plot(self.lbfgs_l1_history)
        plt.title("LBFGS Lambda1 History")
        # plt.savefig("./lbfgs_l1_history.jpg")
        # plt.clf()
        plt.subplot(1,3,3)
        plt.plot(self.lbfgs_l2_history)
        plt.title("LBFGS Lambda2 History")
        plt.savefig("./lbfgs.jpg")
        plt.clf()


    def predict(self, x_star, t_star):
        
        tf_dict = {self.x_u_tf: x_star, self.t_u_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, p_star
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__": 
      
    N_train = 5000
    
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load Data
    x = np.load("./x.npy") # x coordinates
    t = np.load("./t.npy") # time values
    p = np.load("./p.npy") # pressure values
    u = np.load("./u.npy") # velocity values
    # xc = np.load("./xc.npy") # x coordinates of collocation points
    # tc = np.load("./tc.npy") # time values of collocation points

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

    lower_bound_vessel_1=0
    upper_bound_vessel_1=0.04964
    lower_bound_t = t.min()
    upper_bound_t = t.max()
    N_f=5000
    xc = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1)*np.random.random((N_f))[:,None]
    tc = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]

    # Training Data
    xu = np.concatenate([xi,xb], 0)   
    tu = np.concatenate([ti,tb], 0) 
    pu = np.concatenate([pi,pb], 0)
    uu = np.concatenate([ui,ub], 0)
    datau = np.concatenate([xu,tu,pu,uu], 1)
    np.random.shuffle(datau)
    xu = datau[:,0:1]
    tu = datau[:,1:2]
    pu = datau[:,2:3]
    uu = datau[:,3:4]
    x_f = np.concatenate([xu,xc], 0) 
    t_f = np.concatenate([tu,tc], 0)
    np.random.shuffle(x_f)
    np.random.shuffle(t_f)

    # pu = pu/1060


    # Training
    model = BloodViscoVelocityPINN(xu, tu, x_f, t_f, pu, uu, layers)
    model.train(1000)
    
    # Prediction
    u_pred, p_pred = model.predict(x_f, t_f)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    
    # Error

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.000033669)/0.01 * 100

    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))

    # plotting

    # import pdb; pdb.set_trace()

    idx = np.argsort(tu, 0)   
    t = tu[idx,:].flatten()
    p = pu[idx,:].flatten()
    u = uu[idx,:].flatten()
    up = u_pred[idx].flatten()
    pp = p_pred[idx].flatten()

    figure(figsize=(15, 15), dpi=80)

    plt.subplot(2,2,1)
    plt.plot(t,p, linestyle = '-', color='blue', label='Exact pressure')
    plt.title("Exact Pressure")
    # plt.savefig('exact_pressure.jpeg')
    # plt.clf()
    plt.subplot(2,2,2)
    plt.plot(t,pp, linestyle = '-', color='red', label='Predicted pressure')
    plt.title("Predicted Pressure")
    # plt.savefig('predicted_pressure.jpeg')
    # plt.clf()
    plt.subplot(2,2,3)
    plt.plot(t,u, linestyle = '-', color='blue', label='Exact velocity')
    plt.title("Exact Velocity")
    # plt.savefig('exact_velocity.jpeg')
    # plt.clf()
    plt.subplot(2,2,4)
    plt.plot(t,up, linestyle = '-', color='red', label='Predicted velocity')
    plt.title("Predicted Velocity")
    plt.savefig('predicted_exact.jpeg')
    plt.clf()

    