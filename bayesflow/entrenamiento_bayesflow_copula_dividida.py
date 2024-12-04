from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import bayesflow.diagnostics as diag
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.api as sm
import copy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mvn, gamma as xgamma, norm,multivariate_normal
from scipy.special import gamma, factorial
from timeit import default_timer as timer




def simular_logAR1(n, phi, sigma):
   observaciones = np.zeros(n)
   ce = -(sigma**2)/(2*(1-phi**2))
   observaciones[0] =np.exp(np.random.normal(ce, sigma/np.sqrt((1-phi**2))))
   for t in range(1, n):
       #print(observaciones[t-1])
       observaciones[t] = np.exp( (1-phi)*ce + phi * np.log(observaciones[t-1]) + np.random.normal(0, sigma)   )
       #observaciones[t] = (1-phi) +phi * observaciones[t-1] + np.random.normal(0, sigma)
   return observaciones
def simular_X1(n, beta1):
 X1 = (np.random.exponential(scale=1,size=n)**beta1)/gamma(1+beta1)
 return X1
np.random.seed(1)
nsites=25
m=100
n_epochs = 25
n_iterations_per_epoch = 2000
n_batch_size = 128
 


x = [0,0.25,0.5,0.75,1]
y = [0,0.25,0.5,0.75,1]
xv,yv = np.meshgrid(x, y)
NT = np.product(xv.shape)
loc1=np.reshape(xv,NT)
loc2=np.reshape(yv,NT)
loc=np.column_stack((loc1,loc2))  # matriz de diseño (dimensiones: nsites x 4)
Z1 = loc[:, 0]  # primera covariable espacial
Z2 = loc[:, 1]  # segunda covariable espacial
Z3 = np.random.randn(nsites)  # tercera covariable espacial
cov = np.column_stack((np.ones(nsites), Z1, Z2, Z3))  # matriz de diseño (dimensiones: nsites x 4)
# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(loc))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
np.save('dist_mat_uniforme.npy', dist_mat) # save
rho_upper_range = 2*np.max(dist_mat)

def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return X3
def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 3, ncov)
 return simul_gamma

def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)







def model_prior():
    """Generates random draws from uniform pior with rejection sampling."""


    y_train_beta3_auxiliar = np.random.uniform(3,15,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,2*np.max(dist_mat),size=1)[0]

    y_train_auxiliar = [y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
    
    previas = np.array(y_train_auxiliar)
    return(previas)


prior = Prior(prior_fun=model_prior, param_names=[r"$\phi$", r"$\sigma$", r"$\beta_3$", r"$\rho$", r"$\gamma_0$",r"$\gamma_1$",r"$\gamma_2$",r"$\gamma_3$"])
prior_means, prior_stds = prior.estimate_means_and_stds()



def proceso_para_conv2d(params, m):
    y_train_beta1_auxiliar=0.5
    y_train_phi_auxiliar = np.random.uniform(0,1,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #y_train_beta1_auxiliar = 0.5#np.random.uniform(0,1,size=1)[0]
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    #y_train_beta3_auxiliar = np.max([y_train_beta3_auxiliar, 5])  # Restringimos la preva
    y_train_phi_auxiliar = np.min([y_train_phi_auxiliar, 0.95])  # Restringimos la previa para evitar indefiniciones
    y_train_phi_auxiliar = np.max([y_train_phi_auxiliar, 0.05])  # Restringimos la previa para evitar indefiniciones


    
    y_train_beta3_auxiliar,y_train_rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        X_train_auxiliar[sitio] = covariables_auxiliar*X1_auxiliar*X2_auxiliar*X3_auxiliar
        X1_auxiliar_division=simular_X1(m,y_train_beta1_auxiliar)
        X2_auxiliar_division=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X_train_auxiliar[sitio] = X_train_auxiliar[sitio]/(covariables_auxiliar*X1_auxiliar_division*X2_auxiliar_division)
        
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        X_train_auxiliar_para_convolucion.append(X_train_auxiliar[0][tiempo,:].reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=100
simulator2 = Simulator(simulator_fun=partial(proceso_para_conv2d, m=time_points))
model = GenerativeModel(prior, simulator2, name="simulador_proceso")
data = model(batch_size=1000)
sim_mean = np.mean(data["sim_data"])
sim_std = np.std(data["sim_data"])

nombre_modelo = 'copula-dividida-Conv2d-256-LSTM_2_512-DENSE_1_256'
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM(tf.keras.Model):
    def __init__(self, hidden_size=512, summary_dim=512):
        super().__init__()
        timesteps = 100
        features = 20
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((100,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                #TimeDistributed(BatchNormalization()),
                #TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
                #TimeDistributed(BatchNormalization()),
                #TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
                TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )

    def call(self, x, **kwargs):
        #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
        out = self.LSTM(x)
        return out
    
summary_net = CustomLSTM()


COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}



inference_net = InvertibleNetwork(num_params=2, num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer = AmortizedPosterior(inference_net, summary_net, name=nombre_modelo)
trainer = Trainer(amortizer=amortizer, generative_model=model, memory=True, checkpoint_path = nombre_modelo)
history = trainer.train_online(epochs=n_epochs, iterations_per_epoch=n_iterations_per_epoch, batch_size=n_batch_size,early_stopping=True,validation_sims=1000)


valid_sim_data_raw = model(batch_size=500)
valid_sim_data = trainer.configurator(valid_sim_data_raw)
posterior_samples = amortizer.sample(valid_sim_data, n_samples=100)
fig = diag.plot_recovery(posterior_samples, valid_sim_data["parameters"], param_names=prior.param_names)
fig.savefig(nombre_modelo + '.PNG')
