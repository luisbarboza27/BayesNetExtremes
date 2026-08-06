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
def model_prior():
    """Generates random draws from uniform pior with rejection sampling."""
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    previas = y_train_gamma_auxiliar

    return(previas)

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


def proceso(params, m):
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2,high=15,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_beta1_auxiliar = np.random.uniform(0.05,2,size=1)[0]
    y_train_beta2_auxiliar = np.random.uniform(0.05,2,size=1)[0]
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))
    
    Indicator  = np.zeros((nsites,m))
    X2_auxiliar=simular_X1(m,y_train_beta2_auxiliar)
    for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X1_auxiliar*X2_auxiliar*X3_auxiliar*covariables_auxiliar
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

    # agregar parámetros auxiliares
    X_train_auxiliar = np.vstack([
        X_train_auxiliar,
        np.repeat(y_train_beta1_auxiliar, m),
        np.repeat(y_train_beta2_auxiliar, m),
        np.repeat(y_train_beta3_auxiliar, m),
        np.repeat(y_train_rho_auxiliar, m)
    ])

    X_train_auxiliar = X_train_auxiliar.reshape(
        1,
        2*nsites+4,
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
        features =  2*nsites+4
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
        x = tf.reshape(x, (-1, time_points,  2*nsites+4))  # Ajusta según sea necesario
        out = self.LSTM(x)
        return out


COUPLING_NET_SETTINGS = {
    # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
    "num_dense": 2,
    "dropout_prob": 0.2, "bins" : 32
}

import copy  # Necesario para no alterar permanentemente el bloque original en cada iteración

n_epochs = 25
n_iterations_per_epoch = 1000
n_batch_size = 128
block_size = 12800

def ajuste_df_covariable(simul, cov, covariable_quitar):
    for sm in range(len(simul['sim_data'])):
        valor_previa = simul['prior_draws'][sm, covariable_quitar]
        for sitio in range(nsites):
            simul['sim_data'][sm][0][0][:,sitio] = simul['sim_data'][sm][0][0][:,sitio] * quitar_covariable(cov[sitio,(covariable_quitar):(covariable_quitar+1)], [valor_previa])
    
    simul['prior_draws'] = simul['prior_draws'][:, 0:(covariable_quitar)]
    return simul



def entrenar_multiples_modelos_covariables(modelos_config, cov_original, parametros_originales, total_sims=n_iterations_per_epoch * n_batch_size):
    
    trainers = {}
    amortizers = {}
    
    # 1. Inicializar todos los modelos con sus propios amortizers y trainers
    print("Inicializando redes neuronales...")
    for nombre_modelo, cfg in modelos_config.items():
        print(f"Inicializando {nombre_modelo} con {cfg['num_covs']} covariables")
        num_covs = cfg['num_covs']
        
        summary_net = CustomLSTM(cfg['hidden_size'], cfg['summary_dim'])
        inference_net = InvertibleNetwork(
            num_params=num_covs,
            num_coupling_layers=4,
            coupling_settings=COUPLING_NET_SETTINGS,
            coupling_design='spline'
        )
        
        amortizer = AmortizedPosterior(
            inference_net,
            summary_net,
            name=nombre_modelo
        )
        
        trainer = Trainer(
            amortizer=amortizer,
            memory=False,
            checkpoint_path=nombre_modelo
        )
        
        trainers[nombre_modelo] = trainer
        amortizers[nombre_modelo] = amortizer

    inicio = datetime.now()
    print("######################################################################")
    print("Iniciando entrenamiento paso a paso por bloques...")
    
    # 2. Iterar sobre los bloques de simulación
    for i in range(0, total_sims, block_size):
        print(f"\n--- Generando Bloque {i//block_size + 1}: {i} → {i+block_size} ---")
        
        # Generar un bloque de simulaciones completo (con todas las covariables para M7)
        sim_block_base = model(batch_size=block_size)
        sim_block_validation_base = model(batch_size=128)  # Para validación, si es necesario
        
        # Variables de trabajo para ir reduciendo secuencialmente
        sim_block_actual = copy.deepcopy(sim_block_base)
        sim_block_validation_actual = copy.deepcopy(sim_block_validation_base)
        cov_actual = copy.deepcopy(cov_original)
        parametros_actuales = copy.deepcopy(parametros_originales)
        
        # Entrenar cada modelo usando el mismo bloque (ajustado según sus covariables)
        for nombre_modelo, cfg in modelos_config.items():
            num_covs = cfg['num_covs']
            
            # Reducir el bloque si el modelo actual requiere menos covariables
            while cov_actual.shape[1] > num_covs:
                covariable_quitar = cov_actual.shape[1] - 1
                sim_block_actual = ajuste_df_covariable(sim_block_actual, cov_actual, covariable_quitar)
                sim_block_validation_actual = ajuste_df_covariable(sim_block_validation_actual, cov_actual, covariable_quitar)
                cov_actual = cov_actual[:, 0:covariable_quitar]
                parametros_actuales = parametros_actuales[0:covariable_quitar]

                

            print(f"Entrenando {nombre_modelo} ({num_covs} covariables) con bloque {i//block_size + 1}")
            history = trainers[nombre_modelo].train_offline(
                simulations_dict=sim_block_actual,
                epochs=n_epochs,
                batch_size=n_batch_size,
                early_stopping=True,
                validation_sims=sim_block_validation_actual
            )
            
    fin = datetime.now()
    duracion = fin - inicio
    
    # 3. Validación final y ploteo para cada modelo
    print("\nIniciando validación y gráficos finales...")
    for nombre_modelo, cfg in modelos_config.items():
        num_covs = cfg['num_covs']
        n_row = cfg['n_row']
        
        # Generar data de validación nueva
        valid_sim_data_raw = model(batch_size=512)
        valid_cov = copy.deepcopy(cov_original)
        valid_parametros = copy.deepcopy(parametros_originales)
        
        # Ajustar la data de validación al tamaño correcto de covariables
        while valid_cov.shape[1] > num_covs:
            covariable_quitar = valid_cov.shape[1] - 1
            valid_sim_data_raw = ajuste_df_covariable(valid_sim_data_raw, valid_cov, covariable_quitar)
            valid_cov = valid_cov[:, 0:covariable_quitar]
            valid_parametros = valid_parametros[0:covariable_quitar]

        valid_sim_data = trainers[nombre_modelo].configurator(valid_sim_data_raw)
        posterior_samples = amortizers[nombre_modelo].sample(valid_sim_data, n_samples=100)

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

        # Guardar en TXT
        with open(f"{nombre_modelo}.txt", "a") as f:
            f.write("######################################################################\n")
            f.write(f"Modelo: {nombre_modelo}\n")
            f.write(f"Inicio total: {inicio}\n")
            f.write(f"Fin total: {fin}\n")
            f.write(f"Tiempo de ejecución conjunto: {duracion}\n\n")

    print("######################################################################")
    print("Finaliza el entrenamiento de todos los modelos")
    print(f"Inicio: {inicio}")
    print(f"Fin: {fin}")
    print(f"Tiempo de ejecución total: {duracion}")


# --- CONFIGURACIÓN Y EJECUCIÓN ---
print('Inicia preparación de modelos!')

# Definimos el diccionario con la configuración de cada modelo (en orden descendente)
configuracion_modelos = {
    'covariables_DY_aplicacion_M7': {'num_covs': 7, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 3},
    'covariables_DY_aplicacion_M6': {'num_covs': 6, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 2},
    'covariables_DY_aplicacion_M5': {'num_covs': 5, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 2},
    'covariables_DY_aplicacion_M4': {'num_covs': 4, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 2},
    'covariables_DY_aplicacion_M3': {'num_covs': 3, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 1},
    'covariables_DY_aplicacion_M2': {'num_covs': 2, 'hidden_size': 1024, 'summary_dim': 128, 'n_row': 1}
}

# Ejecutamos la función maestra
entrenar_multiples_modelos_covariables(
    modelos_config=configuracion_modelos,
    cov_original=cov_original,           # Tu array matriz global
    parametros_originales=parametros     # Tu lista de parámetros global (los 7 originales)
)