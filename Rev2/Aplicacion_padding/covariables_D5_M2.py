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
import traceback
from sklearn.preprocessing import MinMaxScaler
# importing datetime module
from datetime import datetime
 
# now is a method in datetime module is
# used to retrieve current data,time
myobj = datetime.now()
 
 
# printing the object itself
print("Object:", myobj)
 
 
 
# Configuración para permitir el uso de toda la GPU disponible
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Hacer visibles todas las GPUs disponibles
    tf.config.set_visible_devices(physical_devices, 'GPU')

    # Configurar para permitir el crecimiento dinámico de la memoria de cada GPU
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print('Gpus detectados!')
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


def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return 1/X3
def previa_covariables(ncov):
 simul_gamma = np.random.normal(0, 3, ncov)
 return simul_gamma
 
def calculo_covariable(fila_locacion,gamma_cov):
 suma = 0
 for z in range(len(gamma_cov)):
   suma+=gamma_cov[z]*fila_locacion[z]    
 return np.exp(suma)





n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
 
 
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
cov_original = np.column_stack((np.ones(nsites), Z1))
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


def simular_X3(n,rho,beta3,nsites,dist_mat):
 Sigma = np.exp(-dist_mat / rho)
 Gauss = multivariate_normal.rvs(mean=np.zeros(nsites), cov=Sigma, size=n)  # simulación de vectores gaussianos multivariados independientes (dimensiones: ntime x nsites)
 X3 = (xgamma.ppf(norm.cdf(Gauss), a=beta3, scale=1) / (beta3 - 1))
 return 1/X3


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
print('Inicia funciones de previa')
def model_prior_covariables():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar

    return(previas)

parametros_covariables = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$"]
prior_covariables = Prior(prior_fun=model_prior_covariables, param_names=parametros_covariables)
prior_means, prior_stds = prior_covariables.estimate_means_and_stds()

def configure_input_covariables(forward_dict):
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



def proceso_covariables(params, m):
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2,high=15,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
   
    #Indicator  = np.zeros((nsites,m))
    
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,0.5)
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X1_auxiliar*X2_auxiliar*X3_auxiliar*covariables_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
                # indicador de valores extremos
        #Indicator[sitio] = np.where(
         #   auxi < cuantil_75,
         #   1,
         #   0)
        
    # concatenar indicadores como nuevas variables
    #X_train_auxiliar = np.concatenate(
    #    (X_train_auxiliar, Indicator),
    #    axis=0
    #)

    # agregar parámetros auxiliares
    X_train_auxiliar = np.vstack([
        X_train_auxiliar,
        np.repeat(y_train_phi_auxiliar, m),
        np.repeat(y_train_sigma_auxiliar, m),
        np.repeat(y_train_beta3_auxiliar, m),
        np.repeat(y_train_rho_auxiliar, m)
    ])

    X_train_auxiliar = X_train_auxiliar.reshape(
        1,
        nsites+4,
        m
    ).transpose(0,2,1)

    return np.array([X_train_auxiliar])
time_points=m


simulator_covariables = Simulator(simulator_fun=partial(proceso_covariables, m=time_points))
model_covariables = GenerativeModel(prior_covariables, simulator_covariables, name="simulador_proceso")


data = model_covariables(batch_size=2)
sim_mean = np.mean(data["sim_data"])
sim_std = np.std(data["sim_data"])

class CustomLSTM_covariables(tf.keras.Model):
    def __init__(self, hidden_size=512, summary_dim=512):
        super().__init__()
        timesteps = time_points
        features =  nsites+4
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
        x = tf.reshape(x, (-1, time_points,  nsites+4))  # Ajusta según sea necesario
        out = self.LSTM(x)
        return out


COUPLING_NET_SETTINGS = {
    # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}

import copy  # Necesario para no alterar permanentemente el bloque original en cada iteración


def ajuste_df_covariable(simul, cov, covariable_quitar):
    for sm in range(len(simul['sim_data'])):
        valor_previa = simul['prior_draws'][sm, covariable_quitar]
        for sitio in range(nsites):
            simul['sim_data'][sm][0][0][:,sitio] = simul['sim_data'][sm][0][0][:,sitio] * quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)], [valor_previa])
    
    simul['prior_draws'] = simul['prior_draws'][:, 0:(covariable_quitar)]
    return simul



def entrenar_multiples_modelos_covariables(sim_block_actual,modelos_config,valid_parametros,ultima,duracion_historica,nombre_modelo):
    

    # 1. Inicializar todos los modelos con sus propios amortizers y trainers
    print("Inicializando redes neuronales...")

    num_covs = modelos_config['num_covs']
    nombre_modelo = nombre_modelo
    n_row = modelos_config['n_row']
    summary_net_covariables = CustomLSTM_covariables(modelos_config['hidden_size'], modelos_config['summary_dim'])
    inference_net_covariables = InvertibleNetwork(
        num_params=num_covs,
        num_coupling_layers=4,
        coupling_settings=COUPLING_NET_SETTINGS,
        coupling_design='spline'
    )
    
    amortizer_covariables = AmortizedPosterior(
        inference_net_covariables,
        summary_net_covariables,
        name=nombre_modelo
    )
    
    trainer_covariables = Trainer(
        amortizer=amortizer_covariables,
        generative_model=model_covariables,
        memory=False,
        checkpoint_path=nombre_modelo
    )
    
    
    inicio = datetime.now()
    print("######################################################################")
    print("Iniciando entrenamiento...")
    history = trainer_covariables.train_offline(
    simulations_dict=sim_block_actual,
    epochs=n_epochs,
    batch_size=n_batch_size,
    early_stopping=True,
    validation_sims=128 
    )
    fin = datetime.now()
    duracion = fin - inicio
    
    if ultima:
        # Generar data de validación nueva
        valid_sim_data_raw = model_covariables(batch_size=128)
        valid_sim_data = trainer_covariables.configurator(valid_sim_data_raw)
        posterior_samples = amortizer_covariables.sample(valid_sim_data, n_samples=100)

        # Generar y guardar gráfico
        fig = diag.plot_recovery(
            posterior_samples,
            valid_sim_data["parameters"],
            param_names=valid_parametros,
            xlabel="Real",
            ylabel="Estimado",
            n_row=n_row
        )
        fig.savefig(nombre_modelo + ".PNG")
        duracion = duracion_historica+duracion
        # Guardar en TXT
        with open(f"{nombre_modelo}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write(f"Modelo: {nombre_modelo}\n")
            f.write(f"Tiempo de ejecución: {duracion}\n\n")
            

    return duracion





# --- CONFIGURACIÓN Y EJECUCIÓN ---
print('Inicia preparación de modelos!')

# Definimos el diccionario con la configuración de cada modelo (en orden descendente)
configuracion_modelos = {
    'covariables_D5_aplicacion_M2_V2': {'num_covs': 2, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 1},
}


n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
block_size = 12800
total_sims = n_batch_size*n_iterations_per_epoch



from datetime import timedelta
duraciones = {
    k: timedelta(0)
    for k in configuracion_modelos
}
for i in range(0, total_sims, block_size):
    print(f"\n--- Generando Bloque {i//block_size + 1}: {i} → {i+block_size} ---")
    # Número real de simulaciones de este bloque
    n_sims_bloque = min(block_size, total_sims - i)
    # Indica si este es el último bloque
    ultima = (i + n_sims_bloque >= total_sims)

    # Generar simulaciones
    cov = cov_original
    simul = model_covariables(batch_size=n_sims_bloque)
    simul_previa = simul
    for k, config in configuracion_modelos.items():
        print('Entrenando modelo:  ' + k )
        # Por defecto, usar todos los parámetros/covariables
        parametros_auxiliar = parametros_covariables
        

        # Número de covariables que utiliza este modelo
        num_covs = config['num_covs']

        if num_covs != 2:
            simul_previa = ajuste_df_covariable(
                simul_previa,
                cov,
                num_covs
            )

            cov = cov_original[:, :num_covs]
            parametros_auxiliar = parametros_covariables[:num_covs]

        else:
            cov = cov_original

        try:
            duraciones[k] = entrenar_multiples_modelos_covariables(
                simul_previa,
                configuracion_modelos[k],
                parametros_auxiliar,
                ultima,
                duraciones.get(k),
                k
            )

        except Exception as e:
            # Guardar el error en TXT
            with open(f"{k}.txt", "a") as f:
                f.write("######################################################################\n")
                f.write(f"Modelo: {k}\n")
                f.write(f"Error: {type(e).__name__}: {e}\n")
                f.write("######################################################################\n\n")
                