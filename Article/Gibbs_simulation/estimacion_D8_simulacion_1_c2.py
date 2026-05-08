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
from metricas import mcmc_summary_statistics
import matplotlib.pyplot as plt
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
nsites=100
n_epochs = 25
n_iterations_per_epoch = 500
n_batch_size = 128
n_iter = 10000


 

 
 
x = np.linspace(0, 1 / (2*np.sqrt(2)), 10)
y = np.linspace(0, 1 / (2*np.sqrt(2)), 10)
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
m=200

print(str(nsites) + ' locaciones')
print(str(m) + ' en tiempo')


 
 
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

parametros = [r"$\gamma_{0}$",r"$\gamma_{{lon}}$",r"$\gamma_{{lat}}$",r"$\gamma_{{ran}}$"]

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

def proceso_D8(params, m):
    y_train_gamma_auxiliar = np.zeros(len(cov[0,:]))
    y_train_beta3_auxiliar =  np.random.uniform(2,high=15,size=1)[0]
    y_train_rho_auxiliar =    np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_gamma_auxiliar = params#[2:]
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites+4, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X1_auxiliar=simular_X1(m,0.5)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar[nsites]=y_train_phi_auxiliar
    X_train_auxiliar[nsites+1]=y_train_sigma_auxiliar
    X_train_auxiliar[nsites+2]=y_train_beta3_auxiliar
    X_train_auxiliar[nsites+3]=y_train_rho_auxiliar    
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites+4, m).transpose(0, 2, 1)
    return(np.array([X_train_auxiliar]))
time_points=m


simulator_D8 = Simulator(simulator_fun=partial(proceso_D8, m=time_points))
model_D8 = GenerativeModel(prior, simulator_D8, name="simulador_proceso")


data_D8 = model_D8(batch_size=2)
sim_mean_D8 = np.mean(data_D8["sim_data"])
sim_std_D8 = np.std(data_D8["sim_data"])


print('Inicia simulacion')
#simul_previa_D8 = model_D8(batch_size=n_iterations_per_epoch*n_batch_size)
print('Termina simulacion')



class CustomLSTM_D8(tf.keras.Model):
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


def funcion_prediccion_covariables(simul_previa_D8,cov,parametros,n_epochs,n_batch_size,hidden_size, summary_dim,nombre_modelo):


    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
    
    summary_net_D8 = CustomLSTM_D8(hidden_size, summary_dim)
    inference_net_D8 = InvertibleNetwork(num_params=cov.shape[1], num_coupling_layers=4, coupling_settings=COUPLING_NET_SETTINGS)
    amortizer_D8 = AmortizedPosterior(inference_net_D8, summary_net_D8, name=nombre_modelo)
    trainer_D8 = Trainer(amortizer=amortizer_D8, generative_model=model_D8, memory=False, checkpoint_path = nombre_modelo)
    valid_sim_data_raw_D8 = model_D8(batch_size=1)
    #valid_sim_data_D8 = trainer_D8.configurator(valid_sim_data_raw_D8)
    return(amortizer_D8,trainer_D8,valid_sim_data_raw_D8)


nombre_modelo='covariables_D8_simulacion_extremo_V1'+'2_15'
amortizer_covariables,trainer_covariables,valid_sim_data_1=funcion_prediccion_covariables('simul_previa_D8',cov,parametros,n_epochs,n_batch_size,128, 128,nombre_modelo)
###################################################################
###################################################################
###################################################################
cov = cov_original
def model_prior_D8():
    """Generates random draws from uniform pior with rejection sampling."""
    #y_train_sigma_auxiliar = np.random.uniform(0.05,0.95,size=1)[0]
    y_train_phi_auxiliar = np.random.uniform(-0.85,0.85,size=1)[0]
    
    y_train_sigma_auxiliar= np.random.uniform(0,3,size=1)[0]#np.random.gamma(shape=2,scale=1,size=K)
    #y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = np.random.uniform(2,high=15,size=1)[0]
    y_train_rho_auxiliar =  np.random.uniform(0,2*np.max(dist_mat),size=1)[0]
    y_train_auxiliar = [y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,
                        y_train_rho_auxiliar]
   
    previas = np.array(y_train_auxiliar)
    return(previas)
 
parametros= [ r"$\phi$", r"$\sigma$",r"$\beta_3$", r"$\rho$"]
prior_D8 = Prior(prior_fun=model_prior_D8, param_names=parametros)
prior_means_D8, prior_stds_D8 = prior_D8.estimate_means_and_stds()
 

##################


def proceso_D8(params, m):
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = params
    X_train_auxiliar = np.zeros((nsites, m))
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    
    X1_auxiliar=simular_X1(m,0.5)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        auxi = np.log(X2_auxiliar*X3_auxiliar*X1_auxiliar)
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    X_train_auxiliar_para_convolucion = []
    for tiempo in range(m):
        matriz_expandida = np.zeros((10, 10))
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(10,10)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(10,10,1).tolist())
    return(np.array(X_train_auxiliar_para_convolucion))
time_points=m

simulator_D8 = Simulator(simulator_fun=partial(proceso_D8, m=time_points))
model_D8 = GenerativeModel(prior_D8, simulator_D8, name="simulador_proceso")
def funcion_prediccion_parametros(simul_previa_D8,cov,parametros,n_epochs,n_batch_size,summary_net_D8,nombre_modelo):


    COUPLING_NET_SETTINGS = {
       # "dense_args": dict(units=128, kernel_regularizer=None, activation="relu"),
        "num_dense": 2,
        "dropout_prob": 0.2, "bins" : 32
    }
    
    
    inference_net_D8 = InvertibleNetwork(num_params=len(parametros), num_coupling_layers=10, coupling_settings=COUPLING_NET_SETTINGS,coupling_design='spline')
    amortizer_D8 = AmortizedPosterior(inference_net_D8, summary_net_D8, name=nombre_modelo)
    trainer_D8 = Trainer(amortizer=amortizer_D8, generative_model=model_D8, memory=False, checkpoint_path = nombre_modelo)
    valid_sim_data_raw_D8 = model_D8(batch_size=1)
    #valid_sim_data_D8 = trainer_D8.configurator(valid_sim_data_raw_D8)
    return(amortizer_D8,trainer_D8,valid_sim_data_raw_D8)
    
class CustomLSTM_D8(tf.keras.Model):
    def __init__(self, hidden_size=1000, summary_dim=2000):
        super().__init__()
        timesteps = m
        self.LSTM = tf.keras.Sequential(
            [   tf.keras.layers.Input((timesteps,10, 10, 1)),
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

nombre_modelo='parametros_D8_simulacion_extremo_V11'+'2_15'
summary_net_D8 = CustomLSTM_D8(128, 128)
amortizer_parametros,trainer_parametros,valid_sim_data_2=funcion_prediccion_parametros('simul_previa_D8',cov,parametros,n_epochs,n_batch_size,summary_net_D8,nombre_modelo)




def datos_para_red1(parametros,nsites,m,simulacion,cov_matriz):
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = parametros
    X_train_auxiliar= np.zeros((nsites+4, m))
    for sitio in range(nsites):
        auxi= simulacion_proceso[0][0][:,sitio]
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar[nsites]=y_train_phi_auxiliar
    X_train_auxiliar[nsites+1]=y_train_sigma_auxiliar
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
        matriz55=X_train_auxiliar[0][tiempo,:].reshape(10,10)
        X_train_auxiliar_para_convolucion.append(matriz55.reshape(10,10,1).tolist())
    return np.array(X_train_auxiliar_para_convolucion)




def proceso_D8_estimacion(params_covariables,params_parametros, nsites, m,cov,dist_mat):
    y_train_gamma_auxiliar = params_covariables
    y_train_phi_auxiliar,y_train_sigma_auxiliar,y_train_beta3_auxiliar,y_train_rho_auxiliar = params_parametros
    X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat)
    X_train_auxiliar = np.zeros((nsites, m))

    X1_auxiliar=simular_X1(m,0.5)
    for sitio in range(nsites):
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        auxi=X2_auxiliar*X3_auxiliar*covariables_auxiliar*X1_auxiliar
        cuantil_75 = np.quantile(auxi,0.75)
        X_train_auxiliar[sitio] = np.where(auxi<cuantil_75,cuantil_75,auxi)
    X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)
    return(np.array([X_train_auxiliar]))





phi_real = 0.5
sigma_real = 1
gammas_real = [np.exp(1),1,1,1]
beta3_real = 5
rho_real = 0.5






param_real = [phi_real,sigma_real,beta3_real,rho_real]
simulacion_proceso = proceso_D8_estimacion(gammas_real,param_real, nsites, m,cov,dist_mat)

phi_pred =[]
sigma_pred =[]
g1_pred =[]
g2_pred =[]
g3_pred =[]
g4_pred =[]
beta3_pred = []
rho_pred =[]

phi_init = 0.7
sigma_init = 1.7
beta3_init = 8
rho_init = 0.75


parametros_input = [phi_init,sigma_init,beta3_init,rho_init]
for k in range(n_iter):
    
    
    input_red_1 = datos_para_red1(parametros_input,nsites,m,simulacion_proceso,cov)
    valid_sim_data_1['sim_data']=input_red_1
    valid_sim_data_1_c = trainer_covariables.configurator(valid_sim_data_1)
    posterior_samples_covariables= amortizer_covariables.sample(valid_sim_data_1_c, n_samples=1)[0]
    posterior_g1,posterior_g2,posterior_g3,posterior_g4 =posterior_samples_covariables
    input_red_2 = datos_para_red2(posterior_samples_covariables,nsites,m,simulacion_proceso,cov)
    
    valid_sim_data_2['sim_data']=np.expand_dims(input_red_2, axis=0)
    valid_sim_data_2_c = trainer_parametros.configurator(valid_sim_data_2)
    posterior_phi,posterior_sigma,posterior_beta3,posterior_rho= amortizer_parametros.sample(valid_sim_data_2_c, n_samples=1)[0]

    if posterior_beta3<1.05:
        posterior_beta3 = 1.05
    if posterior_rho<0:
        posterior_rho = 0.05
    if posterior_sigma<0:
        posterior_sigma=0.05
    if posterior_phi< -0.95:
        posterior_phi=-0.95
    if posterior_phi>0.95:
        posterior_phi=0.95
    
    phi_pred += [posterior_phi]
    sigma_pred += [posterior_sigma]
    beta3_pred += [posterior_beta3]
    rho_pred+= [posterior_rho]
    g1_pred+=[posterior_g1]
    g2_pred+=[posterior_g2]
    g3_pred+=[posterior_g3]
    g4_pred+=[posterior_g4]
    print([k,posterior_phi,posterior_sigma,posterior_beta3,posterior_rho],posterior_g1,posterior_g2,posterior_g3,posterior_g4)
    parametros_input = [posterior_phi,posterior_sigma,posterior_beta3,posterior_rho]


df_simul = pd.DataFrame({'posterior_phi':phi_pred,'posterior_sigma':sigma_pred, 'posterior_beta3':beta3_pred,'posterior_rho':rho_pred,'posterior_g1':g1_pred,'posterior_g2':g2_pred,'posterior_g3':g3_pred,'posterior_g4':g4_pred })

resultado_reales=[phi_real,sigma_real,beta3_real,rho_real]+gammas_real
df_result = mcmc_summary_statistics(df_simul,resultado_reales)

print(df_result)
df_result.to_csv('estimacion_r_simulacion_D8_extremo_V1_V11_c2.csv')
df_simul.to_csv('estimacion_t_simulacion_D8_extremo_V1_V11_c2.csv')
