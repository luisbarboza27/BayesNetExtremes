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
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
 
 
# Configuración para permitir el uso de toda la GPU disponible
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Hacer visibles todas las GPUs disponibles
    tf.config.set_visible_devices(physical_devices, 'GPU')

    # Configurar para permitir el crecimiento dinámico de la memoria de cada GPU
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No se detectaron GPUs.")

 
 
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
nsites=225
n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128



 
x = np.linspace(0, 1, 15)
y = np.linspace(0, 1, 15)
xv,yv = np.meshgrid(x, y)
NT = np.product(xv.shape)
loc1=np.reshape(xv,NT)
loc2=np.reshape(yv,NT)
loc=np.column_stack((loc1,loc2))  # matriz de diseño (dimensiones: nsites x 4)
loc = pd.DataFrame({'lat':loc[:, 0],'lon':loc[:, 1]}).sort_values(['lat','lon'],ascending=[False,True])[['lon','lat']]
Z1 = loc['lon'].values # primera covariable espacial
Z2 = loc['lat'].values   # segunda covariable espacial
Z3 = np.random.randn(nsites)  # tercera covariable espacial



scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.reshape(-1,1))
scaler = MinMaxScaler()
Z3 = scaler.fit_transform(Z3.reshape(-1,1))




# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(loc))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(dist_mat)
m=100

print(str(nsites) + ' locaciones')
print(str(m) + ' en tiempo')


 
 
def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return X3


def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 2, ncov)
 return simul_gamma
 
def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)


def quitar_covariable(fila_locacion,gamma_cov):
    suma = -gamma_cov[0]*fila_locacion[0]    
    return np.exp(suma)



cov = cov_original
def model_prior_DY():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.95,size=1)[0]
    y_train_beta2_auxiliar = np.random.uniform(0.05,0.95,size=1)[0]
    #y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = np.random.uniform(3,high=7.5,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_auxiliar = [y_train_beta2_auxiliar,y_train_beta1_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
   
    previas = np.array(y_train_auxiliar)
    return(previas)
 
parametros= [ r"$\beta_2$",r"$\beta_1$",r"$\beta_3$", r"$\rho$"]
prior_DY = Prior(prior_fun=model_prior_DY, param_names=parametros)
prior_means_DY, prior_stds_DY = prior_DY.estimate_means_and_stds()
 

##################


def proceso_DY(params, m):
    y_train_beta2_auxiliar,y_train_beta1_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    #X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        auxi = np.log(X2_auxiliar*X3_auxiliar*X1_auxiliar)
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)

    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((15, 15))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(15,15)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(15,15,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m
simulator_DY = Simulator(simulator_fun=partial(proceso_DY, m=time_points))
model_DY = GenerativeModel(prior_DY, simulator_DY, name="simulador_proceso")
data_DY = model_DY(batch_size=1)




print('Inicia simulacion')
myobj=datetime.now()
print(myobj)
simul_previa_DY = model_DY(batch_size=n_iterations_per_epoch*n_batch_size)
print('Termina simulacion')
myobj=datetime.now()
print(myobj)




def funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo):


    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
    
    
    inference_net_DY = InvertibleNetwork(num_params=len(parametros), num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,coupling_design='spline')
    amortizer_DY = AmortizedPosterior(inference_net_DY, summary_net_DY, name=nombre_modelo)
    trainer_DY = Trainer(amortizer=amortizer_DY, generative_model=model_DY, memory=False, checkpoint_path = nombre_modelo)
     

    myobj= datetime.now()
    print(myobj)
    
    try:
        history_DY = trainer_DY.train_offline(simulations_dict=simul_previa_DY,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
        valid_sim_data_raw_DY = model_DY(batch_size=512)
        valid_sim_data_DY = trainer_DY.configurator(valid_sim_data_raw_DY)
        posterior_samples_DY = amortizer_DY.sample(valid_sim_data_DY, n_samples=100)
        fig = diag.plot_recovery(posterior_samples_DY, valid_sim_data_DY["parameters"], param_names=parametros)
        fig.savefig(nombre_modelo + '.PNG')
        print('######################################################################')
        print('Finaliza '+ nombre_modelo)
    except:
        print('Falla el '+ nombre_modelo)

    myobj= datetime.now()
    print(myobj)

class CustomLSTM_DY(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,15, 15, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
                TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )

    def call(self, x, **kwargs):
        #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
        out = self.LSTM(x)
        return out


# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V11'
# summary_net_DY = CustomLSTM_DY(1000, 2000)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V12'
# summary_net_DY = CustomLSTM_DY(1024, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V13'
# summary_net_DY = CustomLSTM_DY(128, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V14'
# summary_net_DY = CustomLSTM_DY(1024, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V15'
# summary_net_DY = CustomLSTM_DY(128, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)


# class CustomLSTM_DY(tf.keras.Model):
#     def __init__(self, hidden_size=1000, summary_dim=2000):
#         super().__init__()
#         timesteps = m
#         self.LSTM = tf.keras.Sequential(
#             [   tf.keras.layers.Input((timesteps,15, 15, 1)),
#                 TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
#                 TimeDistributed(tf.keras.layers.Flatten()),
#                 tf.keras.layers.LSTM(hidden_size, return_sequences=True),
#                 tf.keras.layers.LSTM(hidden_size, return_sequences=False),
#                 tf.keras.layers.Dense(hidden_size, activation="relu"),
#                 tf.keras.layers.Dense(summary_dim, activation="elu"),
#             ]
#         )

#     def call(self, x, **kwargs):
#         #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
#         out = self.LSTM(x)
#         return out


# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V21'
# summary_net_DY = CustomLSTM_DY(1000, 2000)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V22'
# summary_net_DY = CustomLSTM_DY(1024, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V23'
# summary_net_DY = CustomLSTM_DY(128, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V24'
# summary_net_DY = CustomLSTM_DY(1024, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V25'
# summary_net_DY = CustomLSTM_DY(128, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)



# class CustomLSTM_DY(tf.keras.Model):
#     def __init__(self, hidden_size=1000, summary_dim=2000):
#         super().__init__()
#         timesteps = m
#         self.LSTM = tf.keras.Sequential(
#             [   tf.keras.layers.Input((timesteps,15, 15, 1)),
#                 TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), padding='same')),
#                 TimeDistributed(tf.keras.layers.Flatten()),
#                 tf.keras.layers.LSTM(hidden_size, return_sequences=True),
#                 tf.keras.layers.LSTM(hidden_size, return_sequences=False),
#                 tf.keras.layers.Dense(hidden_size, activation="relu"),
#                 tf.keras.layers.Dense(summary_dim, activation="elu"),
#             ]
#         )

#     def call(self, x, **kwargs):
#         #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
#         out = self.LSTM(x)
#         return out


# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_31'
# summary_net_DY = CustomLSTM_DY(1000, 2000)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V32'
# summary_net_DY = CustomLSTM_DY(1024, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V33'
# summary_net_DY = CustomLSTM_DY(128, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V34'
# summary_net_DY = CustomLSTM_DY(1024, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V35'
# summary_net_DY = CustomLSTM_DY(128, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)



# class CustomLSTM_DY(tf.keras.Model):
#     def __init__(self, hidden_size=1000, summary_dim=2000):
#         super().__init__()
#         timesteps = m
#         self.LSTM = tf.keras.Sequential(
#             [   tf.keras.layers.Input((timesteps,15, 15, 1)),
#                 TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), padding='same')),
#                 TimeDistributed(tf.keras.layers.Flatten()),
#                 tf.keras.layers.LSTM(hidden_size, return_sequences=True),
#                 tf.keras.layers.LSTM(hidden_size, return_sequences=False),
#                 tf.keras.layers.Dense(summary_dim, activation="elu"),
#             ]
#         )

#     def call(self, x, **kwargs):
#         #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
#         out = self.LSTM(x)
#         return out


# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_41'
# summary_net_DY = CustomLSTM_DY(1000, 2000)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V42'
# summary_net_DY = CustomLSTM_DY(1024, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V43'
# summary_net_DY = CustomLSTM_DY(128, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V44'
# summary_net_DY = CustomLSTM_DY(1024, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V45'
# summary_net_DY = CustomLSTM_DY(128, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)


# class CustomLSTM_DY(tf.keras.Model):
#     def __init__(self, hidden_size=1000, summary_dim=2000):
#         super().__init__()
#         timesteps = m
#         self.LSTM = tf.keras.Sequential([
#         tf.keras.layers.Input((timesteps, 5, 5, 1)),
        
#         TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same')),
#         TimeDistributed(BatchNormalization()),
#         TimeDistributed(ReLU()),
        
#         TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding='same')),
#         TimeDistributed(BatchNormalization()),
#         TimeDistributed(ReLU()),
        
#         TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), padding='same')),
#         TimeDistributed(BatchNormalization()),
#         TimeDistributed(ReLU()),
        
#         TimeDistributed(tf.keras.layers.Flatten()),
        
#         Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True)),
#         Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=False)),
        
#         tf.keras.layers.Dense(hidden_size, activation="relu"),
#         tf.keras.layers.Dense(summary_dim, activation="elu")
#         ])

#     def call(self, x, **kwargs):
#         #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
#         out = self.LSTM(x)
#         return out


# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_c_V1'
# summary_net_DY = CustomLSTM_DY(1000, 2000)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_c_V2'
# summary_net_DY = CustomLSTM_DY(1024, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_c_V3'
# summary_net_DY = CustomLSTM_DY(128, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_c_V4'
# summary_net_DY = CustomLSTM_DY(1024, 128)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)

# ###################################################################
# nombre_modelo='parametros_DY_simulacion_extremo_V4'
# summary_net_DY = CustomLSTM_DY(128, 1024)
# entrenamiento=funcion_entrenamiento(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,summary_net_DY,nombre_modelo)