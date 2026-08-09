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



def gibbs(n_iter,simulacion_proceso,n_covariables,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables):
    print('###################################')
    print('Inicia Gibbs...')
    print('###################################')
    
    resultados = []
    muestras_paralelas=250
    for start_idx in range(0, n_iter,muestras_paralelas):
        valid_sim_data_1['sim_data']=simulacion_proceso
        valid_sim_data_1_c = trainer_covariables.configurator(valid_sim_data_1)
        posterior_samples = amortizer_covariables.sample(valid_sim_data_1_c, n_samples=muestras_paralelas)
        resultados.append(posterior_samples)
        if start_idx%1000==0:
            print('############################################')
            print('Iteracion: '+str(start_idx))
    print('Finaliza Gibbs...')
    resultados = np.vstack(resultados)
    df_covariables = pd.DataFrame(resultados,columns = ['posterior_phi','posterior_sigma']+['posterior_g'+str(z+1) for z in range(n_covariables)])
    df_covariables['posterior_sigma']=np.where(df_covariables['posterior_sigma']<0,0.05,df_covariables['posterior_sigma'])
    df_covariables['posterior_phi']=np.where(df_covariables['posterior_phi']<-0.99,-0.99,df_covariables['posterior_phi'])
    df_covariables['posterior_phi']=np.where(df_covariables['posterior_phi']>0.99,0.99,df_covariables['posterior_phi'])
    return(df_covariables)


datos_guanacaste = pd.read_csv('datosPrecGuanacaste.csv')
datos_guanacaste['lon']=np.round(datos_guanacaste['lon'],3)
datos_guanacaste['lat']=np.round(datos_guanacaste['lat'],3)
datos_guanacaste=datos_guanacaste[datos_guanacaste.date<'2022-09-01']
datos_guanacaste=datos_guanacaste.sort_values('date')
loc = pd.read_csv('covariables_guanacaste.csv')
loc = loc.sort_values(['lat','lon'],ascending=[False,True])[['lon','lat','alt']]
# 1
test_loc = loc.sample(int(len(loc) * 0.25), random_state=777)
# Without test data
locaciones_completas = loc[
    ~pd.MultiIndex.from_frame(loc[['lat', 'lon']]).isin(
        pd.MultiIndex.from_frame(test_loc[['lat', 'lon']])
    )
]
nsites=len(locaciones_completas)
# 1


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
dist_mat = squareform(pdist(locaciones_completas[['lon','lat']]))
cov_original = np.column_stack((np.ones(nsites), Z1, Z2, Z3, Z12, Z22, Z32))
rho_upper_range = 2*np.max(squareform(pdist(loc)))
m=int(len(datos_guanacaste)/len(loc))

print(str(nsites) + ' locations')
print(str(m) + ' in time')


loc_all = (
    pd.MultiIndex.from_product(
        [loc["lat"].unique(), loc["lon"].unique()],
        names=["lat", "lon"]
    )
    .to_frame(index=False)
)
# Use locaciones_completas to ommit test also
locaciones_completas_aux=locaciones_completas
locaciones_completas_aux['aux'] = 1
loc_all_completar = loc_all.merge(locaciones_completas_aux,on = ['lon','lat'],how = 'left').sort_values(['lat','lon'],ascending=[False,True])
nsite_completados = len(loc_all_completar)

datos_guanacaste_filtrados=datos_guanacaste.merge(locaciones_completas[['lon','lat']],how='inner',on =['lon','lat'])


# Para la estimacion de covariables
X_train_guanacaste= np.zeros((nsites, m))
Indicator  = np.zeros((nsites,m))
for sitio in range(nsites):
    lon_actual=locaciones_completas.iloc[sitio].lon
    lat_actual=locaciones_completas.iloc[sitio].lat
    auxi= datos_guanacaste_filtrados[(datos_guanacaste_filtrados.lon==lon_actual)&(datos_guanacaste_filtrados.lat==lat_actual)].sort_values('date')['chirps'].values
    cuantil_75 = np.quantile(auxi,0.75)
    X_train_guanacaste[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    Indicator[sitio] = np.where(
    auxi < cuantil_75,
    1,
    0)

# concatenar indicadores como nuevas variables
X_train_guanacaste = np.concatenate(
    (X_train_guanacaste, Indicator),
    axis=0
)

X_train_guanacaste = X_train_guanacaste.reshape(
    1,
    2*nsites,
    m
).transpose(0,2,1)



df_resultados_final = pd.DataFrame({})

#############
# Red de parametros
############




cov = cov_original
parametros = [r"$\phi$","$\sigma$",r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{alt}}$",r"$\gamma_{{lon}^2}$",r"$\gamma_{{lat}^2}$",r"$\gamma_{{alt}^2}$"]



def funcion_prediccion_covariables(simul_previa_DY,cov,parametros,n_epochs,n_batch_size,hidden_size, summary_dim,nombre_modelo):
    
    def model_prior():
        """Generates random draws from uniform pior with rejection sampling."""
        y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
        previas = y_train_gamma_auxiliar
        y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
        y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
        previas=np.append([y_train_phi_auxiliar,y_train_sigma_auxiliar],previas)
        return(previas)

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



    def proceso(params, m):
        y_train_gamma_auxiliar = params[2:]
        y_train_phi_auxiliar = params[0]
        y_train_sigma_auxiliar = params[1]
        X_train_auxiliar = np.zeros((nsites, m))
        Indicator  = np.zeros((nsites,m))
        
        for sitio in range(nsites):
            covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
            X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
            auxi=X2_auxiliar*covariables_auxiliar
            cuantil_75 = np.quantile(auxi,0.75)
            X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
                    # indicador de valores extremos
            Indicator[sitio] = np.where(
                auxi < cuantil_75,
                1,
                0)
            
        # concatenar indicadores como nuevas variables
        X_train_auxiliar = np.concatenate(
            (X_train_auxiliar, Indicator),
            axis=0
        )

        X_train_auxiliar = X_train_auxiliar.reshape(
            1,
            2*nsites,
            m
        ).transpose(0,2,1)

        return np.array([X_train_auxiliar])
    time_points=m


    simulator = Simulator(simulator_fun=partial(proceso, m=time_points))
    model = GenerativeModel(prior, simulator, name="simulador_proceso")


    data = model(batch_size=2)
    sim_mean = np.mean(data["sim_data"])
    sim_std = np.std(data["sim_data"])

    class CustomLSTM(tf.keras.Model):
        def __init__(self, hidden_size=512, summary_dim=512):
            super().__init__()
            timesteps = time_points
            features =  2*nsites
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
            x = tf.reshape(x, (-1, time_points,  2*nsites))  # Ajusta según sea necesario
            out = self.LSTM(x)
            return out


    COUPLING_NET_SETTINGS = {
        # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
        
        

    summary_net = CustomLSTM(hidden_size, summary_dim)
    inference_net = InvertibleNetwork(num_params=cov.shape[1]+2, num_coupling_layers=4, coupling_settings=COUPLING_NET_SETTINGS)
    amortizer = AmortizedPosterior(inference_net, summary_net, name=nombre_modelo)
    trainer = Trainer(amortizer=amortizer, generative_model=model, memory=False, checkpoint_path = nombre_modelo)
    valid_sim_data_raw = model(batch_size=1)
    #valid_sim_data_DY = trainer_DY.configurator(valid_sim_data_raw_DY)
    return(amortizer,trainer,valid_sim_data_raw)


n_posteriores=2
n_epochs=25
n_batch_size=128
nombre_modelo='covariables_D2_aplicacion_M7'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D2',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)




phi_init = 0.5
sigma_init = 1
simulacion_proceso=tf.expand_dims(X_train_guanacaste, axis=0)
parametros_input = [phi_init,sigma_init]
nombres_parametros = ['posterior_phi','posterior_sigma']
######################################################################################################

trace=gibbs(n_posteriores,simulacion_proceso,7,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=6
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:(covariable_quitar+2)]
nombre_modelo='covariables_D2_aplicacion_M6'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D2',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,simulacion_proceso,covariable_quitar,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables)
##############################################################
covariable_quitar=5
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:(covariable_quitar+2)]
nombre_modelo='covariables_D2_aplicacion_M5'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D2',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,simulacion_proceso,covariable_quitar,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables)
trace.to_csv('trace_'+nombre_modelo+'.csv')

##############################################################
covariable_quitar=4
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:(covariable_quitar+2)]
nombre_modelo='covariables_D2_aplicacion_M4'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D2',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,simulacion_proceso,covariable_quitar,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=3
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:(covariable_quitar+2)]
nombre_modelo='covariables_D2_aplicacion_M3'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D2',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,simulacion_proceso,covariable_quitar,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables)
trace.to_csv('trace_'+nombre_modelo+'.csv')
##############################################################
covariable_quitar=2
cov = cov_original[:,0:covariable_quitar]
parametros=parametros[0:(covariable_quitar+2)]
nombre_modelo='covariables_D2_aplicacion_M2'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D2',cov,parametros,n_epochs,n_batch_size,1024, 128,nombre_modelo)

trace=gibbs(n_posteriores,simulacion_proceso,covariable_quitar,cov,valid_sim_data_1,trainer_covariables,amortizer_covariables)
trace.to_csv('trace_'+nombre_modelo+'.csv')
