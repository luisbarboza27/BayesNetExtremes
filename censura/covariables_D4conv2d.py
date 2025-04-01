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
n_iterations_per_epoch = 500
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
def model_prior_M7():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)

parametros_M7 = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{alt}}$",r"$\gamma_{{lon}^2}$",r"$\gamma_{{lat}^2}$",r"$\gamma_{{alt}^2}$"]

prior_M7 = Prior(prior_fun=model_prior_M7, param_names=parametros_M7)
prior_means_M7, prior_stds_M7 = prior_M7.estimate_means_and_stds()



def proceso_D4_M7(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M7 = Simulator(simulator_fun=partial(proceso_D4_M7, m=time_points))
model_D4_M7 = GenerativeModel(prior_M7, simulator_D4_M7, name="simulador_proceso")


data_D4_M7 = model_D4_M7(batch_size=512)
sim_mean_D4_M7 = np.mean(data_D4_M7["sim_data"])
sim_std_D4_M7 = np.std(data_D4_M7["sim_data"])


print('Inicia entrenamiento!')

###################################################################
nombre_modelo='covariables_D4_M7'
class CustomLSTM_D4_M7(tf.keras.Model):
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
   
summary_net_D4_M7 = CustomLSTM_D4_M7()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
print('Inicia simulacion')
simul_previa_D4_M7 = model_D4_M7(batch_size=n_iterations_per_epoch*n_batch_size)
print('Termina simulacion')



# inference_net_D4_M7 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS)
# amortizer_D4_M7 = AmortizedPosterior(inference_net_D4_M7, summary_net_D4_M7, name=nombre_modelo)
# trainer_D4_M7 = Trainer(amortizer=amortizer_D4_M7, generative_model=model_D4_M7, memory=True, checkpoint_path = nombre_modelo)
# history_D4_M7 = trainer_D4_M7.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
# valid_sim_data_raw_D4_M7 = model_D4_M7(batch_size=512)
# valid_sim_data_D4_M7 = trainer_D4_M7.configurator(valid_sim_data_raw_D4_M7)
# posterior_samples_D4_M7 = amortizer_D4_M7.sample(valid_sim_data_D4_M7, n_samples=100)
# fig = diag.plot_recovery(posterior_samples_D4_M7, valid_sim_data_D4_M7["parameters"], param_names=parametros_M7)
# fig.savefig(nombre_modelo + '.PNG')
# print('######################################################################')
# print('Finaliza '+ nombre_modelo)

covariable_quitar = 6


print('Inicia simulacion')

for sm in range(len(simul_previa_D4_M7['sim_data'])):
    valor_previa = simul_previa_D4_M7['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_D4_M7['sim_data'][sm][tiempo]=simul_previa_D4_M7['sim_data'][sm][tiempo]*matriz55

simul_previa_D4_M7['prior_draws'] = simul_previa_D4_M7['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M6():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)


prior_M6 = Prior(prior_fun=model_prior_M6, param_names=parametros_M7[0:covariable_quitar])
prior_means_M6, prior_stds_M6 = prior_M6.estimate_means_and_stds()



def proceso_D4_M6(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M6 = Simulator(simulator_fun=partial(proceso_D4_M6, m=time_points))
model_D4_M6 = GenerativeModel(prior_M6, simulator_D4_M6, name="simulador_proceso")


data_D4_M6 = model_D4_M6(batch_size=512)
sim_mean_D4_M6 = np.mean(data_D4_M6["sim_data"])
sim_std_D4_M6 = np.std(data_D4_M6["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_D4_M6'
class CustomLSTM_D4_M6(tf.keras.Model):
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
   
summary_net_D4_M6 = CustomLSTM_D4_M6()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



# inference_net_D4_M6 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
# amortizer_D4_M6 = AmortizedPosterior(inference_net_D4_M6, summary_net_D4_M6, name=nombre_modelo)
# trainer_D4_M6 = Trainer(amortizer=amortizer_D4_M6, generative_model=model_D4_M6, memory=True, checkpoint_path = nombre_modelo)
# history_D4_M6 = trainer_D4_M6.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
# valid_sim_data_raw_D4_M6 = model_D4_M6(batch_size=512)
# valid_sim_data_D4_M6 = trainer_D4_M6.configurator(valid_sim_data_raw_D4_M6)
# posterior_samples_D4_M6 = amortizer_D4_M6.sample(valid_sim_data_D4_M6, n_samples=100)
# fig = diag.plot_recovery(posterior_samples_D4_M6, valid_sim_data_D4_M6["parameters"], param_names=parametros_M7[0:covariable_quitar])
# fig.savefig(nombre_modelo + '.PNG')
# print('######################################################################')
# print('Finaliza '+ nombre_modelo)

covariable_quitar = 5 

print('Inicia simulacion')

for sm in range(len(simul_previa_D4_M7['sim_data'])):
    valor_previa = simul_previa_D4_M7['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_D4_M7['sim_data'][sm][tiempo]=simul_previa_D4_M7['sim_data'][sm][tiempo]*matriz55

simul_previa_D4_M7['prior_draws'] = simul_previa_D4_M7['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M5():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)

prior_M5 = Prior(prior_fun=model_prior_M5, param_names=parametros_M7[0:covariable_quitar])
prior_means_M5, prior_stds_M5 = prior_M5.estimate_means_and_stds()



def proceso_D4_M5(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M5 = Simulator(simulator_fun=partial(proceso_D4_M5, m=time_points))
model_D4_M5 = GenerativeModel(prior_M5, simulator_D4_M5, name="simulador_proceso")


data_D4_M5 = model_D4_M5(batch_size=512)
sim_mean_D4_M5 = np.mean(data_D4_M5["sim_data"])
sim_std_D4_M5 = np.std(data_D4_M5["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_D4_M5'
class CustomLSTM_D4_M5(tf.keras.Model):
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
   
summary_net_D4_M5 = CustomLSTM_D4_M5()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



# inference_net_D4_M5 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
# amortizer_D4_M5 = AmortizedPosterior(inference_net_D4_M5, summary_net_D4_M5, name=nombre_modelo)
# trainer_D4_M5 = Trainer(amortizer=amortizer_D4_M5, generative_model=model_D4_M5, memory=True, checkpoint_path = nombre_modelo)
# history_D4_M5 = trainer_D4_M5.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
# valid_sim_data_raw_D4_M5 = model_D4_M5(batch_size=512)
# valid_sim_data_D4_M5 = trainer_D4_M5.configurator(valid_sim_data_raw_D4_M5)
# posterior_samples_D4_M5 = amortizer_D4_M5.sample(valid_sim_data_D4_M5, n_samples=100)
# fig = diag.plot_recovery(posterior_samples_D4_M5, valid_sim_data_D4_M5["parameters"], param_names=parametros_M7[0:covariable_quitar])
# fig.savefig(nombre_modelo + '.PNG')
# print('######################################################################')
# print('Finaliza '+ nombre_modelo)

covariable_quitar = 4 

print('Inicia simulacion')

for sm in range(len(simul_previa_D4_M7['sim_data'])):
    valor_previa = simul_previa_D4_M7['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_D4_M7['sim_data'][sm][tiempo]=simul_previa_D4_M7['sim_data'][sm][tiempo]*matriz55

simul_previa_D4_M7['prior_draws'] = simul_previa_D4_M7['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M4():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)


prior_M4 = Prior(prior_fun=model_prior_M4, param_names=parametros_M7[0:covariable_quitar])
prior_means_M4, prior_stds_M4 = prior_M4.estimate_means_and_stds()



def proceso_D4_M4(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M4 = Simulator(simulator_fun=partial(proceso_D4_M4, m=time_points))
model_D4_M4 = GenerativeModel(prior_M4, simulator_D4_M4, name="simulador_proceso")


data_D4_M4 = model_D4_M4(batch_size=512)
sim_mean_D4_M4 = np.mean(data_D4_M4["sim_data"])
sim_std_D4_M4 = np.std(data_D4_M4["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_D4_M4'
class CustomLSTM_D4_M4(tf.keras.Model):
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
   
summary_net_D4_M4 = CustomLSTM_D4_M4()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



# inference_net_D4_M4 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
# amortizer_D4_M4 = AmortizedPosterior(inference_net_D4_M4, summary_net_D4_M4, name=nombre_modelo)
# trainer_D4_M4 = Trainer(amortizer=amortizer_D4_M4, generative_model=model_D4_M4, memory=True, checkpoint_path = nombre_modelo)
# history_D4_M4 = trainer_D4_M4.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
# valid_sim_data_raw_D4_M4 = model_D4_M4(batch_size=512)
# valid_sim_data_D4_M4 = trainer_D4_M4.configurator(valid_sim_data_raw_D4_M4)
# posterior_samples_D4_M4 = amortizer_D4_M4.sample(valid_sim_data_D4_M4, n_samples=100)
# fig = diag.plot_recovery(posterior_samples_D4_M4, valid_sim_data_D4_M4["parameters"], param_names=parametros_M7[0:covariable_quitar])
# fig.savefig(nombre_modelo + '.PNG')
# print('######################################################################')
# print('Finaliza '+ nombre_modelo)

covariable_quitar = 3

print('Inicia simulacion')

for sm in range(len(simul_previa_D4_M7['sim_data'])):
    valor_previa = simul_previa_D4_M7['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_D4_M7['sim_data'][sm][tiempo]=simul_previa_D4_M7['sim_data'][sm][tiempo]*matriz55

simul_previa_D4_M7['prior_draws'] = simul_previa_D4_M7['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M3():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)


prior_M3 = Prior(prior_fun=model_prior_M3, param_names=parametros_M7[0:covariable_quitar])
prior_means_M3, prior_stds_M3 = prior_M3.estimate_means_and_stds()



def proceso_D4_M3(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M3 = Simulator(simulator_fun=partial(proceso_D4_M3, m=time_points))
model_D4_M3 = GenerativeModel(prior_M3, simulator_D4_M3, name="simulador_proceso")


data_D4_M3 = model_D4_M3(batch_size=512)
sim_mean_D4_M3 = np.mean(data_D4_M3["sim_data"])
sim_std_D4_M3 = np.std(data_D4_M3["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_D4_M3'
class CustomLSTM_D4_M3(tf.keras.Model):
    def __init__(self, hidden_size=100, summary_dim=100):
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
   
summary_net_D4_M3 = CustomLSTM_D4_M3()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_D4_M3 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_D4_M3 = AmortizedPosterior(inference_net_D4_M3, summary_net_D4_M3, name=nombre_modelo)
trainer_D4_M3 = Trainer(amortizer=amortizer_D4_M3, generative_model=model_D4_M3, memory=True, checkpoint_path = nombre_modelo)
history_D4_M3 = trainer_D4_M3.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4_M3 = model_D4_M3(batch_size=512)
valid_sim_data_D4_M3 = trainer_D4_M3.configurator(valid_sim_data_raw_D4_M3)
posterior_samples_D4_M3 = amortizer_D4_M3.sample(valid_sim_data_D4_M3, n_samples=100)
fig = diag.plot_recovery(posterior_samples_D4_M3, valid_sim_data_D4_M3["parameters"], param_names=parametros_M7[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)


covariable_quitar = 2 

print('Inicia simulacion')

for sm in range(len(simul_previa_D4_M7['sim_data'])):
    valor_previa = simul_previa_D4_M7['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_D4_M7['sim_data'][sm][tiempo]=simul_previa_D4_M7['sim_data'][sm][tiempo]*matriz55

simul_previa_D4_M7['prior_draws'] = simul_previa_D4_M7['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M2():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)


prior_M2 = Prior(prior_fun=model_prior_M2, param_names=parametros_M7[0:covariable_quitar])
prior_means_M2, prior_stds_M2 = prior_M2.estimate_means_and_stds()



def proceso_D4_M2(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
   
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M2 = Simulator(simulator_fun=partial(proceso_D4_M2, m=time_points))
model_D4_M2 = GenerativeModel(prior_M2, simulator_D4_M2, name="simulador_proceso")


data_D4_M2 = model_D4_M2(batch_size=512)
sim_mean_D4_M2 = np.mean(data_D4_M2["sim_data"])
sim_std_D4_M2 = np.std(data_D4_M2["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_D4_M2'
class CustomLSTM_D4_M2(tf.keras.Model):
    def __init__(self, hidden_size=100, summary_dim=100):
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
   
summary_net_D4_M2 = CustomLSTM_D4_M2()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_D4_M2 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_D4_M2 = AmortizedPosterior(inference_net_D4_M2, summary_net_D4_M2, name=nombre_modelo)
trainer_D4_M2 = Trainer(amortizer=amortizer_D4_M2, generative_model=model_D4_M2, memory=True, checkpoint_path = nombre_modelo)
history_D4_M2 = trainer_D4_M2.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4_M2 = model_D4_M2(batch_size=512)
valid_sim_data_D4_M2 = trainer_D4_M2.configurator(valid_sim_data_raw_D4_M2)
posterior_samples_D4_M2 = amortizer_D4_M2.sample(valid_sim_data_D4_M2, n_samples=100)
fig = diag.plot_recovery(posterior_samples_D4_M2, valid_sim_data_D4_M2["parameters"], param_names=parametros_M7[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)

covariable_quitar = 1

print('Inicia simulacion')

for sm in range(len(simul_previa_D4_M7['sim_data'])):
    valor_previa = simul_previa_D4_M7['prior_draws'][sm,covariable_quitar]
    X_train_auxiliar = np.zeros((nsites, 1))
    for sitio in range(nsites):
        X_train_auxiliar[sitio] = quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)],[valor_previa])
    matriz55=X_train_auxiliar.reshape(5,5)
    matriz55=np.array(matriz55.reshape(5,5,1).tolist())    
    for tiempo in range(m):
        simul_previa_D4_M7['sim_data'][sm][tiempo]=simul_previa_D4_M7['sim_data'][sm][tiempo]*matriz55

simul_previa_D4_M7['prior_draws'] = simul_previa_D4_M7['prior_draws'][:,0:(covariable_quitar)]
print('Termina simulacion')





cov = cov_original[:,0:covariable_quitar]
print('Inicia funciones de previa')
def model_prior_M1():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar
    #y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    #y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
    return(previas)


prior_M1 = Prior(prior_fun=model_prior_M1, param_names=parametros_M7[0:covariable_quitar])
prior_means_M1, prior_stds_M1 = prior_M1.estimate_means_and_stds()



def proceso_D4_M1(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2.5,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(0.05,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar
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


simulator_D4_M1 = Simulator(simulator_fun=partial(proceso_D4_M1, m=time_points))
model_D4_M1 = GenerativeModel(prior_M1, simulator_D4_M1, name="simulador_proceso")


data_D4_M1 = model_D4_M1(batch_size=512)
sim_mean_D4_M1 = np.mean(data_D4_M1["sim_data"])
sim_std_D4_M1 = np.std(data_D4_M1["sim_data"])


print('Inicia entrenamiento!')

###################################################################

nombre_modelo='covariables_D4_M1'
class CustomLSTM_D4_M1(tf.keras.Model):
    def __init__(self, hidden_size=100, summary_dim=100):
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
   
summary_net_D4_M1 = CustomLSTM_D4_M1()
 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 



inference_net_D4_M1 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS)
amortizer_D4_M1 = AmortizedPosterior(inference_net_D4_M1, summary_net_D4_M1, name=nombre_modelo)
trainer_D4_M1 = Trainer(amortizer=amortizer_D4_M1, generative_model=model_D4_M1, memory=True, checkpoint_path = nombre_modelo)
history_D4_M1 = trainer_D4_M1.train_offline(simulations_dict=simul_previa_D4_M7,epochs=n_epochs,batch_size=n_batch_size,early_stopping=True,validation_sims=128)
valid_sim_data_raw_D4_M1 = model_D4_M1(batch_size=512)
valid_sim_data_D4_M1 = trainer_D4_M1.configurator(valid_sim_data_raw_D4_M1)
posterior_samples_D4_M1 = amortizer_D4_M1.sample(valid_sim_data_D4_M1, n_samples=100)
fig = diag.plot_recovery(posterior_samples_D4_M1, valid_sim_data_D4_M1["parameters"], param_names=parametros_M7[0:covariable_quitar])
fig.savefig(nombre_modelo + '.PNG')
print('######################################################################')
print('Finaliza '+ nombre_modelo)
