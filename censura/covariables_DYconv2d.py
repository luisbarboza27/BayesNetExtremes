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
nsites=25
n_epochs = 25
n_iterations_per_epoch = 250
n_batch_size = 128



 
datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste.sort_values('date')
loc = datos_guanacaste.groupby(['lat','lon'],as_index=False).agg(casa = ('id','count')).sort_values(['lat','lon'],ascending=[False,True])[['lon','lat']]
# xv,yv = np.meshgrid(np.unique(loc['lon']), np.unique(loc['lat']))
# NT = np.product(xv.shape)
# loc1=np.reshape(xv,NT)
# loc2=np.reshape(yv,NT)
# loca=np.column_stack((loc1,loc2))  # matriz de diseño (dimensiones: nsites x 4)
# locaciones_completas=pd.DataFrame({'lon_c':loca[:,0],'lat_c':loca[:,1]}).sort_values(['lat_c','lon_c'],ascending=[False,True])
locaciones_completas=loc[(loc['lon']>-85.7) & (loc['lon']<-85.2)  & (loc['lat']>10.3) & (loc['lat']<10.8)]

import requests

def obtener_elevacion(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    response = requests.get(url)
    data = response.json()
    
    if 'results' in data:
        return data['results'][0]['elevation']
    else:
        return None

altitud = [235.0, 76.0, 447.0, 628.0, 583.0, 136.0, 55.0, 144.0, 262.0, 358.0, 113.0, 33.0, 118.0, 115.0, 122.0, 160.0, 17.0, 7.0, 46.0, 43.0, 26.0, 15.0, 35.0, 132.0, 4.0]
# Obtener elevación para cada coordenada
# for (index, row) in locaciones_completas[['lat','lon']].iterrows():
#     lat=row['lat']
#     lon=row['lon']
    
#     elevacion = obtener_elevacion(lat, lon)
#     altitud +=[elevacion]
    
#     print(f"La elevación de ({lat}, {lon}) es {elevacion} metros.")



Z1 = locaciones_completas['lon']  # primera covariable espacial
Z2 = locaciones_completas['lat']   # segunda covariable espacial
Z12 = locaciones_completas['lon']**2  # primera covariable espacial
Z22 = locaciones_completas['lat']**2   # segunda covariable espacial
Z3 = np.array(altitud)
Z32 = np.array(altitud)**2







scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))
scaler = MinMaxScaler()
Z12 = scaler.fit_transform(Z12.values.reshape(-1,1))
scaler = MinMaxScaler()
Z22 = scaler.fit_transform(Z22.values.reshape(-1,1))
scaler = MinMaxScaler()
Z3 = scaler.fit_transform(Z3.reshape(-1,1))
scaler = MinMaxScaler()
Z32 = scaler.fit_transform(Z32.reshape(-1,1))




# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(locaciones_completas))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(dist_mat)
m=int(len(datos_guanacaste)/len(loc))

print(str(nsites) + ' locaciones')
print(str(m) + ' en tiempo')



 
 
def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return X3


def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 1, ncov)
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
print('Inicia funciones de previa')
def model_prior_M7Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)

parametros_M7Y = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{alt}}$",r"$\gamma_{{lon}^2}$",r"$\gamma_{{lat}^2}$",r"$\gamma_{{alt}^2}$"]

prior_M7Y = Prior(prior_fun=model_prior_M7Y, param_names=parametros_M7Y)
prior_means_M7Y, prior_stds_M7Y = prior_M7Y.estimate_means_and_stds()



def proceso_DY_M7Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M7Y = Simulator(simulator_fun=partial(proceso_DY_M7Y, m=time_points))
model_DY_M7Y = GenerativeModel(prior_M7Y, simulator_DY_M7Y, name="simulador_proceso")


data_DY_M7Y = model_DY_M7Y(batch_size=512)
sim_mean_DY_M7Y = np.mean(data_DY_M7Y["sim_data"])
sim_std_DY_M7Y = np.std(data_DY_M7Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################
nombre_modelo='covariables_DY_M7Y'
class CustomLSTM_DY_M7Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M7Y = CustomLSTM_DY_M7Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
print('Inicia simulacion')
simul_previa_DY_M7Y = model_DY_M7Y(batch_size=n_iterations_per_epoch*n_batch_size)
print('Termina simulacion')


def configure_input(forward_dict):
    """Configures dictionary of prior draws and simulated data into BayesFlow format."""

    out_dict = {}

    # standardization sim_data
    sim_data = forward_dict["sim_data"].astype(np.float32)
    #norm_data = (sim_data - sim_mean) / sim_std

    # standardization priors
    params = forward_dict["prior_draws"].astype(np.float32)
    #norm_params = (params - prior_means) / prior_stds

    # remove nan, inf and -inf
    keep_idx = np.all(np.isfinite(sim_data), axis=(1, 2,3,4))
    if not np.all(keep_idx):
        print("Invalid value encountered...removing from batch")

    # add to dict
    out_dict["summary_conditions"] = sim_data[keep_idx]
    out_dict["parameters"] = params[keep_idx]

    return out_dict


# inference_net_DY_M7Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS)
# amortizer_DY_M7Y = AmortizedPosterior(inference_net_DY_M7Y, summary_net_DY_M7Y, name=nombre_modelo)
# trainer_DY_M7Y = Trainer(amortizer=amortizer_DY_M7Y, generative_model=model_DY_M7Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
# history_DY_M7Y = trainer_DY_M7Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
# valid_sim_data_raw_DY_M7Y = model_DY_M7Y(batch_size=512)
# valid_sim_data_DY_M7Y = trainer_DY_M7Y.configurator(valid_sim_data_raw_DY_M7Y)
# posterior_samples_DY_M7Y = amortizer_DY_M7Y.sample(valid_sim_data_DY_M7Y, n_samples=100)
# fig = diag.plot_recovery(posterior_samples_DY_M7Y, valid_sim_data_DY_M7Y["parameters"], param_names=parametros_M7Y)
# fig.savefig(nombre_modelo + '.PNG')
# print('######################################################################')
# print('Finaliza '+ nombre_modelo)

covariable_quitar = 6


print('Inicia simulacion')

for sm in range(len(simul_previa_DY_M7Y['sim_data'])):
    valor_previa = simul_previa_DY_M7Y['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_DY_M7Y['sim_data'][sm][tiempo]=simul_previa_DY_M7Y['sim_data'][sm][tiempo]*matriz55

simul_previa_DY_M7Y['prior_draws'] = simul_previa_DY_M7Y['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M6Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)


prior_M6Y = Prior(prior_fun=model_prior_M6Y, param_names=parametros_M7Y[0:covariable_quitar])
prior_means_M6Y, prior_stds_M6Y = prior_M6Y.estimate_means_and_stds()



def proceso_DY_M6Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M6Y = Simulator(simulator_fun=partial(proceso_DY_M6Y, m=time_points))
model_DY_M6Y = GenerativeModel(prior_M6Y, simulator_DY_M6Y, name="simulador_proceso")


data_DY_M6Y = model_DY_M6Y(batch_size=512)
sim_mean_DY_M6Y = np.mean(data_DY_M6Y["sim_data"])
sim_std_DY_M6Y = np.std(data_DY_M6Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_DY_M6Y'
class CustomLSTM_DY_M6Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M6Y = CustomLSTM_DY_M6Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



# inference_net_DY_M6Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
# amortizer_DY_M6Y = AmortizedPosterior(inference_net_DY_M6Y, summary_net_DY_M6Y, name=nombre_modelo)
# trainer_DY_M6Y = Trainer(amortizer=amortizer_DY_M6Y, generative_model=model_DY_M6Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
# history_DY_M6Y = trainer_DY_M6Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
# valid_sim_data_raw_DY_M6Y = model_DY_M6Y(batch_size=512)
# valid_sim_data_DY_M6Y = trainer_DY_M6Y.configurator(valid_sim_data_raw_DY_M6Y)
# posterior_samples_DY_M6Y = amortizer_DY_M6Y.sample(valid_sim_data_DY_M6Y, n_samples=100)
# fig = diag.plot_recovery(posterior_samples_DY_M6Y, valid_sim_data_DY_M6Y["parameters"], param_names=parametros_M7Y[0:covariable_quitar])
# fig.savefig(nombre_modelo + '.PNG')
# print('######################################################################')
# print('Finaliza '+ nombre_modelo)

covariable_quitar = 5 

print('Inicia simulacion')

for sm in range(len(simul_previa_DY_M7Y['sim_data'])):
    valor_previa = simul_previa_DY_M7Y['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_DY_M7Y['sim_data'][sm][tiempo]=simul_previa_DY_M7Y['sim_data'][sm][tiempo]*matriz55

simul_previa_DY_M7Y['prior_draws'] = simul_previa_DY_M7Y['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M5Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)

prior_M5Y = Prior(prior_fun=model_prior_M5Y, param_names=parametros_M7Y[0:covariable_quitar])
prior_means_M5Y, prior_stds_M5Y = prior_M5Y.estimate_means_and_stds()



def proceso_DY_M5Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M5Y = Simulator(simulator_fun=partial(proceso_DY_M5Y, m=time_points))
model_DY_M5Y = GenerativeModel(prior_M5Y, simulator_DY_M5Y, name="simulador_proceso")


data_DY_M5Y = model_DY_M5Y(batch_size=512)
sim_mean_DY_M5Y = np.mean(data_DY_M5Y["sim_data"])
sim_std_DY_M5Y = np.std(data_DY_M5Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_DY_M5Y'
class CustomLSTM_DY_M5Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M5Y = CustomLSTM_DY_M5Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_DY_M5Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_DY_M5Y = AmortizedPosterior(inference_net_DY_M5Y, summary_net_DY_M5Y, name=nombre_modelo)
trainer_DY_M5Y = Trainer(amortizer=amortizer_DY_M5Y, generative_model=model_DY_M5Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
history_DY_M5Y = trainer_DY_M5Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_DY_M5Y = model_DY_M5Y(batch_size=512)
valid_sim_data_DY_M5Y = trainer_DY_M5Y.configurator(valid_sim_data_raw_DY_M5Y)
posterior_samples_DY_M5Y = amortizer_DY_M5Y.sample(valid_sim_data_DY_M5Y, n_samples=100)
fig = diag.plot_recovery(posterior_samples_DY_M5Y, valid_sim_data_DY_M5Y["parameters"], param_names=parametros_M7Y[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)

covariable_quitar = 4 

print('Inicia simulacion')

for sm in range(len(simul_previa_DY_M7Y['sim_data'])):
    valor_previa = simul_previa_DY_M7Y['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_DY_M7Y['sim_data'][sm][tiempo]=simul_previa_DY_M7Y['sim_data'][sm][tiempo]*matriz55

simul_previa_DY_M7Y['prior_draws'] = simul_previa_DY_M7Y['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M4Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)


prior_M4Y = Prior(prior_fun=model_prior_M4Y, param_names=parametros_M7Y[0:covariable_quitar])
prior_means_M4Y, prior_stds_M4Y = prior_M4Y.estimate_means_and_stds()



def proceso_DY_M4Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M4Y = Simulator(simulator_fun=partial(proceso_DY_M4Y, m=time_points))
model_DY_M4Y = GenerativeModel(prior_M4Y, simulator_DY_M4Y, name="simulador_proceso")


data_DY_M4Y = model_DY_M4Y(batch_size=512)
sim_mean_DY_M4Y = np.mean(data_DY_M4Y["sim_data"])
sim_std_DY_M4Y = np.std(data_DY_M4Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_DY_M4Y'
class CustomLSTM_DY_M4Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M4Y = CustomLSTM_DY_M4Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_DY_M4Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_DY_M4Y = AmortizedPosterior(inference_net_DY_M4Y, summary_net_DY_M4Y, name=nombre_modelo)
trainer_DY_M4Y = Trainer(amortizer=amortizer_DY_M4Y, generative_model=model_DY_M4Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
history_DY_M4Y = trainer_DY_M4Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_DY_M4Y = model_DY_M4Y(batch_size=512)
valid_sim_data_DY_M4Y = trainer_DY_M4Y.configurator(valid_sim_data_raw_DY_M4Y)
posterior_samples_DY_M4Y = amortizer_DY_M4Y.sample(valid_sim_data_DY_M4Y, n_samples=100)
fig = diag.plot_recovery(posterior_samples_DY_M4Y, valid_sim_data_DY_M4Y["parameters"], param_names=parametros_M7Y[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)

covariable_quitar = 3

print('Inicia simulacion')

for sm in range(len(simul_previa_DY_M7Y['sim_data'])):
    valor_previa = simul_previa_DY_M7Y['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_DY_M7Y['sim_data'][sm][tiempo]=simul_previa_DY_M7Y['sim_data'][sm][tiempo]*matriz55

simul_previa_DY_M7Y['prior_draws'] = simul_previa_DY_M7Y['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M3Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)


prior_M3Y = Prior(prior_fun=model_prior_M3Y, param_names=parametros_M7Y[0:covariable_quitar])
prior_means_M3Y, prior_stds_M3Y = prior_M3Y.estimate_means_and_stds()



def proceso_DY_M3Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M3Y = Simulator(simulator_fun=partial(proceso_DY_M3Y, m=time_points))
model_DY_M3Y = GenerativeModel(prior_M3Y, simulator_DY_M3Y, name="simulador_proceso")


data_DY_M3Y = model_DY_M3Y(batch_size=512)
sim_mean_DY_M3Y = np.mean(data_DY_M3Y["sim_data"])
sim_std_DY_M3Y = np.std(data_DY_M3Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_DY_M3Y'
class CustomLSTM_DY_M3Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M3Y = CustomLSTM_DY_M3Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_DY_M3Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_DY_M3Y = AmortizedPosterior(inference_net_DY_M3Y, summary_net_DY_M3Y, name=nombre_modelo)
trainer_DY_M3Y = Trainer(amortizer=amortizer_DY_M3Y, generative_model=model_DY_M3Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
history_DY_M3Y = trainer_DY_M3Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_DY_M3Y = model_DY_M3Y(batch_size=512)
valid_sim_data_DY_M3Y = trainer_DY_M3Y.configurator(valid_sim_data_raw_DY_M3Y)
posterior_samples_DY_M3Y = amortizer_DY_M3Y.sample(valid_sim_data_DY_M3Y, n_samples=100)
fig = diag.plot_recovery(posterior_samples_DY_M3Y, valid_sim_data_DY_M3Y["parameters"], param_names=parametros_M7Y[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)


covariable_quitar = 2 

print('Inicia simulacion')

for sm in range(len(simul_previa_DY_M7Y['sim_data'])):
    valor_previa = simul_previa_DY_M7Y['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_DY_M7Y['sim_data'][sm][tiempo]=simul_previa_DY_M7Y['sim_data'][sm][tiempo]*matriz55

simul_previa_DY_M7Y['prior_draws'] = simul_previa_DY_M7Y['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M2Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)


prior_M2Y = Prior(prior_fun=model_prior_M2Y, param_names=parametros_M7Y[0:covariable_quitar])
prior_means_M2Y, prior_stds_M2Y = prior_M2Y.estimate_means_and_stds()



def proceso_DY_M2Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M2Y = Simulator(simulator_fun=partial(proceso_DY_M2Y, m=time_points))
model_DY_M2Y = GenerativeModel(prior_M2Y, simulator_DY_M2Y, name="simulador_proceso")


data_DY_M2Y = model_DY_M2Y(batch_size=512)
sim_mean_DY_M2Y = np.mean(data_DY_M2Y["sim_data"])
sim_std_DY_M2Y = np.std(data_DY_M2Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_DY_M2Y'
class CustomLSTM_DY_M2Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M2Y = CustomLSTM_DY_M2Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_DY_M2Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_DY_M2Y = AmortizedPosterior(inference_net_DY_M2Y, summary_net_DY_M2Y, name=nombre_modelo)
trainer_DY_M2Y = Trainer(amortizer=amortizer_DY_M2Y, generative_model=model_DY_M2Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
history_DY_M2Y =trainer_DY_M2Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_DY_M2Y = model_DY_M2Y(batch_size=512)
valid_sim_data_DY_M2Y = trainer_DY_M2Y.configurator(valid_sim_data_raw_DY_M2Y)
posterior_samples_DY_M2Y = amortizer_DY_M2Y.sample(valid_sim_data_DY_M2Y, n_samples=100)
fig = diag.plot_recovery(posterior_samples_DY_M2Y, valid_sim_data_DY_M2Y["parameters"], param_names=parametros_M7Y[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)

covariable_quitar = 1

print('Inicia simulacion')

for sm in range(len(simul_previa_DY_M7Y['sim_data'])):
    valor_previa = simul_previa_DY_M7Y['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_DY_M7Y['sim_data'][sm][tiempo]=simul_previa_DY_M7Y['sim_data'][sm][tiempo]*matriz55

simul_previa_DY_M7Y['prior_draws'] = simul_previa_DY_M7Y['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M1Y():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_beta2_auxiliar,y_train_beta1_auxiliar],previas)
    return(previas)


prior_M1Y = Prior(prior_fun=model_prior_M1Y, param_names=parametros_M7Y[0:covariable_quitar])
prior_means_M1Y, prior_stds_M1Y = prior_M1Y.estimate_means_and_stds()



def proceso_DY_M1Y(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,0.85,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m


simulator_DY_M1Y = Simulator(simulator_fun=partial(proceso_DY_M1Y, m=time_points))
model_DY_M1Y = GenerativeModel(prior_M1Y, simulator_DY_M1Y, name="simulador_proceso")


data_DY_M1Y = model_DY_M1Y(batch_size=512)
sim_mean_DY_M1Y = np.mean(data_DY_M1Y["sim_data"])
sim_std_DY_M1Y = np.std(data_DY_M1Y["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_DY_M1Y'
class CustomLSTM_DY_M1Y(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
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
   
summary_net_DY_M1Y = CustomLSTM_DY_M1Y()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_DY_M1Y = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS)
amortizer_DY_M1Y = AmortizedPosterior(inference_net_DY_M1Y, summary_net_DY_M1Y, name=nombre_modelo)
trainer_DY_M1Y = Trainer(amortizer=amortizer_DY_M1Y, generative_model=model_DY_M1Y, memory=True, checkpoint_path = nombre_modelo,configurator=configure_input)
history_DY_M1Y = trainer_DY_M1Y.train_offline(simulations_dict=simul_previa_DY_M7Y,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_DY_M1Y = model_DY_M1Y(batch_size=512)
valid_sim_data_DY_M1Y = trainer_DY_M1Y.configurator(valid_sim_data_raw_DY_M1Y)
posterior_samples_DY_M1Y = amortizer_DY_M1Y.sample(valid_sim_data_DY_M1Y, n_samples=100)
fig = diag.plot_recovery(posterior_samples_DY_M1Y, valid_sim_data_DY_M1Y["parameters"], param_names=parametros_M7Y[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)
