# Load required libraries
import tensorflow as tf
import numpy as np
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1234)
tf.set_random_seed(1234)

# The main solver class
class BloodViscoVelocityPINN:
    # Initialize the class
    def __init__(self, X_measurement_aorta1,
                       T_measurement, T_initial, 
                       X_initial,
                       X_inference_aorta1, 
                       T_inference,
                       P_initial_aorta1,
                       P_boundary_aorta1,
                       X_residual_aorta1, 
                       T_residual,layers,bif_points):

        self.pi = np.pi

        # Reference vessel areas
        self.A_01 = 0.0002734605
        
        # Blood density and viscosity
        self.lambda_1 = tf.Variable([1060.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([3.5], dtype=tf.float32)

        # characteristic velocity                  
        self.U = 0.30038288

        # Charactiristic variables for non-dimensionalization
        self.L = np.sqrt(self.A_01)
        self.T = self.L/self.U
        # l1 = self.lambda_1
        l1 = 1060
        self.p0 = l1*self.U**2 

        # print("h", self.lambda_1*self.U**2)       

        # Non-dimensionalize
        self.A0 = self.L**2     
        
        X_measurement_aorta1 = X_measurement_aorta1/self.L
        X_residual_aorta1 = X_residual_aorta1/self.L
        X_inference = X_inference_aorta1/self.L
        X_initial = X_initial/self.L
        T_measurement  = T_measurement/self.T
        T_residual  = T_residual/self.T
        T_inference = T_inference/self.T
        T_initial  = T_initial/self.T
        
        # Normalize inputs
        self.Xmean1, self.Xstd1 = X_residual_aorta1.mean(0), X_residual_aorta1.std(0)
        self.Tmean, self.Tstd = T_residual.mean(0), T_residual.std(0)
        
        # Jacobians
        self.jac_x1 = 1.0/self.Xstd1
        self.jac_t = 1.0/self.Tstd
        
        # Store normalized/non-dimensionalized variables
        self.X_f1 = (X_residual_aorta1 - self.Xmean1)/self.Xstd1
        self.X_u1 = (X_measurement_aorta1 - self.Xmean1)/self.Xstd1
        self.X_i = (X_inference - self.Xmean1)/self.Xstd1
        self.X_ini = (X_initial - self.Xmean1)/self.Xstd1
        self.T_u = (T_measurement - self.Tmean)/self.Tstd
        self.T_f = (T_residual - self.Tmean)/self.Tstd
        self.T_i = (T_inference - self.Tmean)/self.Tstd
        self.T_ini = (T_initial - self.Tmean)/self.Tstd
        
        self.P_u1 = P_boundary_aorta1*1e-5
        self.P_ini = P_initial_aorta1*1e-5
        
        X1_fm = bif_points[0]/self.L
        bif_p1 = (X1_fm - self.Xmean1)/self.Xstd1        
        X1max = bif_p1[0]
        
        # Store neural net layer dimensions
        self.layers = layers
        
        # Initialize network weights and biases        
        self.weights1, self.biases1 = self.initialize_NN(layers)
        self.weights2, self.biases2 = self.initialize_NN(layers)
        self.weights3, self.biases3 = self.initialize_NN(layers)
        self.weights4, self.biases4 = self.initialize_NN(layers)
                       
        # Define placeholders and computational graph
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        self.X1_fm = tf.constant([X1max], shape = [1024,1], dtype=tf.float32)
        self.P_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.P_u1.shape[1]))
        self.P_ini_tf1 = tf.placeholder(tf.float32, shape=(None, self.P_ini.shape[1]))
        self.X_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.P_ini.shape[1]))
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, self.P_ini.shape[1]))
        self.X_i_tf = tf.placeholder(tf.float32,  shape=(None, self.X_i.shape[1]))
        self.t_i_tf = tf.placeholder(tf.float32,  shape=(None, self.T_i.shape[1]))
        self.X_f_tf1 = tf.placeholder(tf.float32, shape=(None, self.X_f1.shape[1]))
        self.t_f_tf = tf.placeholder(tf.float32, shape=(None, self.T_f.shape[1]))
        self.X_ini_tf1 = tf.placeholder(tf.float32, shape=(None, self.X_ini.shape[1]))
        self.t_ini_tf = tf.placeholder(tf.float32, shape=(None, self.T_ini.shape[1]))
        
        # Neural net predictions
        self.A_u_pred1, self.u_u_pred1, self.P_u_pred1  = self.neural_net_aorta1(self.X_u_tf1, self.t_u_tf)
        self.A_ui_pred1, self.u_ui_pred1, self.P_ui_pred1  = self.neural_net_aorta1(self.X_i_tf, self.t_i_tf)
        self.A_f_pred1, self.u_f_pred1, self.p_f_pred1  = self.neural_net_aorta1(self.X_f_tf1, self.t_f_tf)
        self.A_ini_pred1, self.u_ini_pred1, self.p_ini_pred1  = self.neural_net_aorta1(self.X_ini_tf1, self.t_ini_tf)

        
        
        # Compute PDE residuals
        self.r_A1, self.r_u1, self.r_p1  = self.pinn_aorta1(self.X_f_tf1, self.t_f_tf)
            
        # Compute loss functions
        self.loss_P1, self.loss_P2 = self.compute_measurement_loss_aorta1(self.P_u_pred1, self.p_ini_pred1)
        self.loss_rA1, self.loss_ru1, self.loss_rp1 = self.compute_residual_loss_aorta1(self.r_A1, self.r_u1, self.r_p1)
        
        self.loss_interface  = self.compute_interface_loss()
        
        self.loss_measurements = self.loss_P1 + self.loss_P2
        
        self.loss_ru = self.loss_ru1
        self.loss_rA = self.loss_rA1
        self.loss_rp = self.loss_rp1
        self.loss_residual = self.loss_rA + self.loss_ru + self.loss_rp
        
        # Total loss
        self.loss = self.loss_interface + self.loss_residual  + self.loss_measurements
        
        # Define optimizer        
        self.optimizer  = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = False

        # Define Tensorflow session
        self.sess = tf.Session(config=config)
        
        # Initialize Tensorflow variables
        self.merged = tf.summary.merge_all()
        
        # Logger
        self.summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)
        self.saver = tf.train.Saver()
        self.loss_value_log = [] 
        self.loss_P_log  = []
        self.loss_r_log = []
        self.loss_rp_log = []
        self.loss_c_log = []
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
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
       
    # Neural net forward pass       
    def neural_net(self, H, weights, biases, layers):
        num_layers = len(layers)  
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_aorta1(self, x, t):
        Aup = self.neural_net(tf.concat([x,t],1),self.weights1,self.biases1,self.layers)
        A = Aup[:,0:1]
        u = Aup[:,1:2]
        p = Aup[:,2:3]
        # print(u)
        # print(p)
        return tf.exp(A), u, p

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
     
    # Compute residuals
    def pinn_aorta1(self, x, t):

        l1 = self.lambda_1
        l2 = self.lambda_2
        
        A, u, p = self.neural_net_aorta1(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_01 = self.get_equilibrium_cross_sectional_area_aorta_1(x)
        beta1 = self.get_beta_aorta_1(x)
        
        r_p  = 10000. + beta1*(tf.sqrt(A*self.A0) - tf.sqrt(A_01)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x1

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x1
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x1
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + 0.11*u*u*u_x + (0.1*u*u*A_x)/A + p_x/l1 + (22*l2*self.pi*u)/A
        
        return r_A, r_u, r_p
    
    def get_beta_aorta_1(self, x):
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

    # Compute residual losses
    def compute_residual_loss_aorta1(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))
        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred1 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp
    
    def compute_interface_loss(self):
        
         A1, u1, p1 = self.neural_net_aorta1(self.X_u_tf1,self.t_u_tf) # A*, u*, p*
         A2, u2, p2 = self.neural_net_aorta1(self.X_i_tf,self.t_i_tf) # A*, u*, p*
         Q1 = A1*u1
         Q2 = A2*u2
         loss_mass = tf.reduce_mean(tf.square(Q1-Q2)) 
         p_1 = p1 + (0.5*self.lambda_1*u1**2)
         p_2 = p2 + (0.5*self.lambda_1*u1**2)
         loss_press = tf.reduce_mean(tf.square(p_1-p_2))
                             
         return  loss_mass + loss_press

    def compute_measurement_loss_aorta1(self, p_u, p_ini):
    
        loss_P1 = tf.reduce_mean(tf.square(self.P_u1 - p_u*(1/self.p0)))
        loss_P2 = tf.reduce_mean(tf.square(self.P_ini - p_ini*(1/self.p0)))

        return loss_P1, loss_P2
      
    # Fetch a mini-batch of data for stochastic gradient updates
    def fetch_minibatch(self, X1_f, t_f, N_f_batch):        
        N_f = X1_f.shape[0]
        idx_f = np.random.choice(N_f, N_f_batch, replace=False)
        X1_f_batch = X1_f[idx_f,:]
        t_f_batch = t_f[idx_f,:]    

        return  X1_f_batch, t_f_batch
             
    # Trains the model by minimizing the MSE loss using mini-batch stochastic gradient descent
    def train(self, nIter = 10000, batch_size = 1024, learning_rate = 1e-3): 

        start_time = timeit.default_timer()
        for it in tqdm(range(nIter)):
            
            # Fetch a mini-batch of training data
            # X1_f_batch, T_f_batch = self.fetch_minibatch(self.X_f1, 
            #                                              self.T_f,
            #                                              N_f_batch = batch_size)
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_u_tf1: self.X_u1,
                       self.X_i_tf: self.X_i,
                       self.X_f_tf1: self.X_f1,
                       self.X_ini_tf1: self.X_ini,
                       self.t_f_tf:  self.T_f, 
                       self.t_u_tf:  self.T_u,
                       self.t_i_tf:  self.T_i,
                       self.t_ini_tf: self.T_ini,
                       self.P_u_tf1: self.P_u1, 
                       self.P_ini_tf1: self.P_ini,
                       self.learning_rate: learning_rate}
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # print(self.u_f_pred1)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value, loss_P, loss_r, loss_rp, loss_c, lr, lm  = self.sess.run([self.loss, 
                                                                                      self.loss_P1 + self.loss_P2,
                                                                                      self.loss_ru+self.loss_rA, 
                                                                                      self.loss_rp, 
                                                                                      self.loss_interface,
                                                                                      self.loss_residual,
                                                                                      self.loss_measurements], tf_dict)
                self.loss_value_log.append(loss_value) 
                self.loss_P_log.append(loss_P) 
                self.loss_r_log.append(loss_r) 
                self.loss_rp_log.append(loss_rp) 
                self.loss_c_log.append(loss_c) 
                print('It: %d, Total Loss: %.3e, Loss_residue: %.3e, Loss_interface: %.3e, Loss_measurements: %.3e Time: %.2f' % 
                      (it, loss_value, loss_c, lr, lm, elapsed))
#                 start_time = timeit.default_timer()
                                
    # Evaluates predictions at test points           
    def predict_aorta1(self, X1, t): 
        # non-dimensionalize inputs
        X1 = X1/self.L
        t  = t/self.T
        # normalize inputs
        X1 = (X1 - self.Xmean1)/self.Xstd1
        t = (t - self.Tmean)/self.Tstd
        # Create tf dictionary
        tf_dict1 = {self.X_f_tf1: X1, self.t_f_tf: t}    
        # Evaluate predictions
        A_star1 = self.sess.run(self.A_f_pred1, tf_dict1) 
        u_star1 = self.sess.run(self.u_f_pred1, tf_dict1) 
        p_star1 = self.sess.run(self.p_f_pred1, tf_dict1) 
        # de-normalize outputs        
        A_star1 = A_star1*self.A0
        u_star1 = u_star1*self.U
        p_star1 = p_star1*self.p0

        # print(self.U)
        # print("j")
        # print(self.p0)
        # print(self.lambda_1)
              
        return A_star1, u_star1, p_star1


if __name__ == "__main__": 
    # Define the number of spatio-temporal domain points to evaluate the residual
    # of the system of equations.

    N_f =  2000

    data = np.load("./results_real_PINNs.npy", allow_pickle=True).item()
    aorta1_velocity = data["Velocity_aorta1"]
    aorta1_area = data["Area_aorta1"]
    aorta1_pressure = data["Pressure_aorta1"]
    t = data['Time']

    velocity_measurements_aorta1 = aorta1_velocity
    area_measurements_aorta1 = aorta1_area
    presure_measurements_aorta1 = aorta1_pressure

    # Number of measurements

    N_u = t.shape[0]

    layers = [2, 100, 100, 100, 100, 100, 100, 3]

    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)

    lower_bound_vessel_1 = 0.0   
    upper_bound_vessel_1 = 0.04964

    # Spatial/temporal coordinates
    X_initial_aorta1 = np.linspace(lower_bound_vessel_1,upper_bound_vessel_1,N_u)[:,None]
    T_initial  = lower_bound_t*np.ones((N_u))[:,None]
    X_inference_aorta1 = upper_bound_vessel_1*np.ones((N_u))[:,None]
    T_inference = t
    X_boundary_aorta1 = lower_bound_vessel_1*np.ones((N_u))[:,None]
    T_boundary = t
    X_residual_aorta1 = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1)*np.random.random((N_f))[:,None]
    T_residual = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]

    P_initial_aorta1 = np.min(presure_measurements_aorta1)*np.ones((N_u))[:,None]
    P_boundary_aorta1 = presure_measurements_aorta1
    bif_points = [upper_bound_vessel_1]

    # Build the PINN model
    model = BloodViscoVelocityPINN(X_boundary_aorta1,
                       T_boundary, T_initial, 
                       X_initial_aorta1,
                       X_inference_aorta1, 
                       T_inference,
                       P_initial_aorta1,
                       P_boundary_aorta1,
                       X_residual_aorta1, 
                       T_residual,layers,bif_points)
    
    # Train the PINN model using mini-batch stochastic gradient descent
    model.train(nIter = 1000, batch_size = 1024, learning_rate = 1e-2)

    # model.eval()
    A_predict_aorta1, u_predict_aorta1, p_predict_aorta1     = model.predict_aorta1(X_boundary_aorta1, T_boundary)

    plt.subplot(2,2,1)
    plt.plot(T_boundary,presure_measurements_aorta1, linestyle = '-', color='blue', label='Exact pressure')
    plt.title("Exact Pressure")
    # plt.savefig('exact_pressure.jpeg')
    # plt.clf()
    plt.subplot(2,2,2)
    plt.plot(T_boundary,p_predict_aorta1.flatten(), linestyle = '-', color='red', label='Predicted pressure')
    plt.title("Predicted Pressure")
    # plt.savefig('predicted_pressure.jpeg')
    # plt.clf()
    plt.subplot(2,2,3)
    plt.plot(T_boundary,velocity_measurements_aorta1, linestyle = '-', color='blue', label='Exact velocity')
    plt.title("Exact Velocity")
    # plt.savefig('exact_velocity.jpeg')
    # plt.clf()
    plt.subplot(2,2,4)
    plt.plot(T_boundary,u_predict_aorta1.flatten(), linestyle = '-', color='red', label='Predicted velocity')
    plt.title("Predicted Velocity")
    plt.savefig('predicted_exact.jpeg')
    plt.clf()