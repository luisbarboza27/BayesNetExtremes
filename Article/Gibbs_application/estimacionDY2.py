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
import requests
#from calculadora_de_error import calculadora_de_error
 
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

def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return 1/X3


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


def datos_para_red1(parametros,nsites,m,simulacion,cov_matriz):
    y_train_beta2_auxiliar,y_train_beta1_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = parametros
    X_train_auxiliar= np.zeros((nsites+4, m))
    for sitio in range(nsites):
        auxi= simulacion_proceso[0][0][:,sitio]
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar[nsites]=y_train_beta1_auxiliar
    X_train_auxiliar[nsites+1]=y_train_beta2_auxiliar
    X_train_auxiliar[nsites+2]=y_train_beta3_auxiliar
    X_train_auxiliar[nsites+3]=y_train_rho_auxiliar    
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites+4, m).transpose(0, 2, 1)
    return(np.array([X_train_auxiliar]))



def datos_para_red2(covariables,nsites,m,simulacion,cov_matriz):
    X_train_auxiliar= np.zeros((nsites, m))
    for sitio in range(nsites):
        auxi= simulacion_proceso[0][0][:,sitio]
        covariables_auxiliar = calculo_covariable(cov_matriz[sitio,:],covariables)
        auxi = np.log(auxi/covariables_auxiliar)
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return np.array(X_train_auxiliar_para_convolucion)


def gibbs(n_iter,nombres_parametros,n_covariables,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_parametros,amortizer_parametros):
    resultados_cov = []
    p1_pred = []
    p2_pred = []
    beta3_pred = []
    rho_pred= []
    for k in range(n_iter):
        input_red_1 = datos_para_red1(parametros_input,nsites,m,simulacion_proceso,cov)
        valid_sim_data_1['sim_data']=input_red_1
        valid_sim_data_1_c = trainer_covariables.configurator(valid_sim_data_1)
        posterior_samples_covariables= amortizer_covariables.sample(valid_sim_data_1_c, n_samples=1)[0]
        input_red_2 = datos_para_red2(posterior_samples_covariables,nsites,m,simulacion_proceso,cov)
        valid_sim_data_raw_DY['sim_data']=np.expand_dims(input_red_2, axis=0)
        valid_sim_data_raw_DY_c = trainer_parametros.configurator(valid_sim_data_raw_DY)
        # Cambio
        p1,p2,posterior_beta3,posterior_rho= amortizer_parametros.sample(valid_sim_data_raw_DY_c, n_samples=1)[0]
    
        if posterior_beta3<1:
            posterior_beta3 = 1.05
        if posterior_rho<0:
            posterior_rho = 0.05
        if p1<0:
            p1=0.05
        if p2<0:
            p2=0.05

        p1_pred += [p1]
        p2_pred += [p2]
        beta3_pred += [posterior_beta3]
        rho_pred+= [posterior_rho]
        print([k,p1,p2,posterior_beta3,posterior_rho],posterior_samples_covariables)
        parametros_input = [p1,p2,posterior_beta3,posterior_rho]
        resultados_cov.append(posterior_samples_covariables)
    df_simul = pd.DataFrame({nombres_parametros[0]:p1_pred,nombres_parametros[1]:p2_pred, nombres_parametros[2]:beta3_pred,nombres_parametros[3]:rho_pred})
    df_covariables = pd.DataFrame(resultados_cov,columns = ['posterior_g'+str(z+1) for z in range(n_covariables)])
    resultado_final = pd.concat([df_covariables,df_simul],axis=1)
    return(resultado_final)


np.random.seed(1)
nsites=25
n_epochs = 25
n_iterations_per_epoch = 250
n_batch_size = 128
n_posteriores=10000


datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste.sort_values('date')
datos_guanacaste=datos_guanacaste[datos_guanacaste.date<'2022-09-01']
 
loc = pd.read_csv('covariables_guanacaste.csv')
locaciones_completas=loc[(loc['lon']>-85.7) & (loc['lon']<-85.2)  & (loc['lat']>10.3) & (loc['lat']<10.8)]
Z1 = locaciones_completas['lon']  # primera covariable espacial
Z2 = locaciones_completas['lat']   # segunda covariable espacial
Z12 = locaciones_completas['lon']**2  # primera covariable espacial
Z22 = locaciones_completas['lat']**2   # segunda covariable espacial
Z3 = locaciones_completas['alt'] 
Z32 = locaciones_completas['alt']**2 


scaler = MinMaxScaler()
Z1 = scaler.fit_transform(Z1.values.reshape(-1,1))
scaler = MinMaxScaler()
Z2 = scaler.fit_transform(Z2.values.reshape(-1,1))
scaler = MinMaxScaler()
Z12 = scaler.fit_transform(Z12.values.reshape(-1,1))
scaler = MinMaxScaler()
Z22 = scaler.fit_transform(Z22.values.reshape(-1,1))
scaler = MinMaxScaler()
Z3 = scaler.fit_transform(Z3.values.reshape(-1,1))
scaler = MinMaxScaler()
Z32 = scaler.fit_transform(Z32.values.reshape(-1,1))




# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(locaciones_completas[['lon','lat']]))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)
#np.save('dist_mat_guanacaste.npy', dist_mat) # save
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))  # matriz de diseño (dimensiones: nsites x 4)
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_guanacaste)/len(loc))


print(str(nsites) + ' locaciones')
print(str(m) + ' en tiempo')



datos_guanacaste_filtrados=datos_guanacaste.merge(locaciones_completas[['lon','lat']],how='inner',on =['lon','lat'])


# Para la estimacion de covariables
X_train_guanacaste= np.zeros((nsites, m))

for sitio in range(nsites):
    lon_actual=locaciones_completas.iloc[sitio].lon
    lat_actual=locaciones_completas.iloc[sitio].lat
    auxi= datos_guanacaste_filtrados[(datos_guanacaste_filtrados.lon==lon_actual)&(datos_guanacaste_filtrados.lat==lat_actual)].sort_values('date')['chirps'].values
    cuantil_75 = np.quantile(auxi,0.75)
    X_train_guanacaste[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    
X_train_guanacaste = X_train_guanacaste.reshape(1, nsites, m).transpose(0, 2, 1)
X_train_auxiliar_para_convolucion = []
for tiempo in range(m):
    matriz_expandida = np.zeros((5, 5))
    matriz55=X_train_guanacaste[0][tiempo,:].reshape(5,5)
    X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
X_train_guanacaste_convolucion=np.array(X_train_auxiliar_para_convolucion)


# cambiar proceso de parametros
# cambiar proceso de covariables

df_resultados_final = pd.DataFrame({})





#############
# Red de parametros
############

def model_prior_DY():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_beta1_auxiliar = np.random.uniform(0.05,2,size=1)[0]
    y_train_beta2_auxiliar = np.random.uniform(0.05,2,size=1)[0]
    #y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = np.random.uniform(2,high=7.5,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_auxiliar = [y_train_beta2_auxiliar,y_train_beta1_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
   
    previas = np.array(y_train_auxiliar)
    return(previas)
 
parametros_DY= [ r"$\beta_2$",r"$\beta_1$",r"$\beta_3$", r"$\rho$"]
prior_DY = Prior(prior_fun=model_prior_DY, param_names=parametros_DY)
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
        matriz_expandida = np.zeros((5, 5))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(5,5)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(5,5,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m
simulator_DY = Simulator(simulator_fun=partial(proceso_DY, m=time_points))
model_DY = GenerativeModel(prior_DY, simulator_DY, name="simulador_proceso")


nombre_modelo = 'parametros_DYbb2'
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense
class CustomLSTM_DY(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        hidden_size_1 = 256
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,5, 5, 1)),
                TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')),
                TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
                TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
                tf.keras.layers.Dense(hidden_size_1, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )

    def call(self, x, **kwargs):
        #x = tf.reshape(x, (-1, 100, 20))  # Ajusta según sea necesario 
        out = self.LSTM(x)
        return out
    
summary_net_DY = CustomLSTM_DY()

 
 
COUPLING_NET_SETTINGS = {
   # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}
 
 

inference_net_DY = InvertibleNetwork(num_params=len(parametros_DY),num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,   coupling_design = 'spline')
amortizer_DY = AmortizedPosterior(inference_net_DY, summary_net_DY, name=nombre_modelo)
trainer_DY = Trainer(amortizer=amortizer_DY, generative_model=model_DY, memory=False, checkpoint_path = nombre_modelo)
valid_sim_data_raw_DY = model_DY(batch_size=1)

#############
# Redes de covariables
############



print('Inicia funciones de previa')
def model_prior():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar

    return(previas)

cov = cov_original
parametros = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{alt}}$",r"$\gamma_{{lon}^2}$",r"$\gamma_{{lat}^2}$",r"$\gamma_{{alt}^2}$"]

prior = Prior(prior_fun=model_prior, param_names=parametros)
prior_means, prior_stds = prior.estimate_means_and_stds()

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


def proceso_DY(params, m):
    #y_train_beta1_auxiliar=np.random.uniform(0.05,0.95,size=1)[0]
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2,high=7.5,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,rho_upper_range,size=1)[0]
    y_train_gamma_auxiliar = params
    y_train_beta1_auxiliar = np.random.uniform(0.05,2,size=1)[0]
    y_train_beta2_auxiliar= np.random.uniform(0.05,2,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites+4, m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar[nsites]=y_train_beta1_auxiliar
    X_train_auxiliar[nsites+1]=y_train_beta2_auxiliar
    X_train_auxiliar[nsites+2]=y_train_beta3_auxiliar
    X_train_auxiliar[nsites+3]=y_train_rho_auxiliar    
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites+4, m).transpose(0, 2, 1)
    return(np.array([X_train_auxiliar]))
time_points=m


simulator_DY = Simulator(simulator_fun=partial(proceso_DY, m=time_points))
model_DY = GenerativeModel(prior, simulator_DY, name="simulador_proceso")


data_DY = model_DY(batch_size=2)
sim_mean_DY = np.mean(data_DY["sim_data"])
sim_std_DY = np.std(data_DY["sim_data"])


print('Inicia simulacion')
#simul_previa_DY = model_DY(batch_size=n_iterations_per_epoch*n_batch_size)
print('Termina simulacion')



class CustomLSTM_DY(tf.keras.Model):
    def __init__(self, hidden_size=512, summary_dim=512):
        super().__init__()
        timesteps = time_points
        features = nsites+4
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps, features)),
                tf.keras.layers.LSTM(hidden_size, return_sequences=True),
                tf.keras.layers.LSTM(hidden_size, return_sequences=False),
 #               tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(summary_dim, activation="elu"),
            ]
        )
 
    def call(self, x, **kwargs):
        x = tf.reshape(x, (-1, time_points, nsites+4))  # Ajusta según sea necesario
        out = self.LSTM(x)
        return out


def funcion_prediccion_covariables(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,hidden_size, summary_dim,nombre_modelo):
    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
    
    summary_net_DY = CustomLSTM_DY(hidden_size, summary_dim)
    inference_net_DY = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=4, coupling_settings=COUPLING_NET_SETTINGS)
    amortizer_DY = AmortizedPosterior(inference_net_DY, summary_net_DY, name=nombre_modelo)
    trainer_DY = Trainer(amortizer=amortizer_DY, generative_model=model_DY, memory=False, checkpoint_path = nombre_modelo)
    valid_sim_data_raw_DY = model_DY(batch_size=1)
    #valid_sim_data_DY = trainer_DY.configurator(valid_sim_data_raw_DY)
    return(amortizer_DY,trainer_DY,valid_sim_data_raw_DY)


nombre_modelo='covariables_DY_aplicacion_M72'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)




beta1_init = 0.5
beta2_init = 0.5
beta3_init = 5
rho_init = 0.5
simulacion_proceso=tf.expand_dims(X_train_guanacaste, axis=0)
parametros_input = [beta2_init,beta1_init,beta3_init,rho_init]
nombres_parametros = ['posterior_beta2','posterior_beta1','posterior_beta3','posterior_rho']
######################################################################################################

trace=gibbs(n_posteriores,nombres_parametros,7,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=6
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_DY_aplicacion_M62'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=5
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_DY_aplicacion_M52'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=4
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_DY_aplicacion_M42'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=3
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_DY_aplicacion_M32'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=2
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_DY_aplicacion_M22'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=1
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:covariable_quitar]
nombre_modelo='covariables_DY_aplicacion_M12'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_DY',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,nombres_parametros,covariable_quitar,parametros_input,nsites,m,simulacion_proceso,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables,valid_sim_data_raw_DY,trainer_DY,amortizer_DY)
trace.to_csv('trace_'+nombre_modelo+'.csv')





