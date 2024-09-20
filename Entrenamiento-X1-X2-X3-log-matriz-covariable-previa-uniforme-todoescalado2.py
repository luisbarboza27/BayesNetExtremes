

# 
# print('demole tiempo al otro de 2h')
# from time import sleep
# #sleep(7200)
# print('finalizo el tiempo de espera')

workers_clus=48
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
a1=simular_logAR1(1000,0.5,3) # Mas de 3 en desviacion da problemas
# pd.DataFrame(a1).describe()
# salida = np.random.gamma(shape=3,scale=0.5,size=10000)
# pd.DataFrame(salida).describe()
# sns.histplot(salida)
# plt.show()
# plt.close()
def simular_X1(n, beta1):
  X1 = np.random.exponential(scale=1,size=n)**beta1/gamma(1+beta1)
  return X1

np.random.seed(1)
K=10000
J=100
m=100
nsites=20


loc = np.random.rand(nsites, 2)  # generación de ubicaciones uniformemente distribuidas en el cuadrado unitario [0,1]^2
Z1 = loc[:, 0]  # primera covariable espacial
Z2 = loc[:, 1]  # segunda covariable espacial
Z3 = np.random.randn(nsites)  # tercera covariable espacial
cov = np.column_stack((np.ones(nsites), Z1, Z2, Z3))  # matriz de diseño (dimensiones: nsites x 4)
# MATRIZ DE DISTANCIA ENTRE NUESTROS SITIOS
dist_mat = squareform(pdist(loc))  # matriz de distancia de todas las ubicaciones (dimensiones: nsites x nsites)

np.save('dist_mat.npy', dist_mat) # save
#print(dist_mat)
rho_upper_range = 2*np.max(dist_mat)



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

# Define tus puntos de control
checkpoints = {10,50,100,1000,2500, 5000,7500,10000,12500, 15000,17500,20000,25000,27500}

def print_progress(itera, checkpoints):
    if itera in checkpoints:
        print(f'Iteración en: {itera}')


#   
# c=simular_X3(100,3,10,nsites,dist_mat)
# (c==0).any()
# pd.DataFrame(c[0,:]).describe()
# a=np.random.gamma(shape=1/3,scale=100,size=1000)
# pd.DataFrame(a).describe()
# Demostracion de problemas al no tocar la previa, en valores muy pequenos no se tiene media uno. En 1 da infinito.

# a=simular_AR1(100, 0.5, 2)
#b1=simular_X1(100, 0.8)#
#b2=simular_X1(100, 0.8)#

#y_train_beta3 = np.repeat(np.random.gamma(shape=1/3,scale=500,size=1000),1)
# pd.DataFrame(y_train_beta3).describe().round(3)


#print(np.mean(1/c[:,1]))

#sm.graphics.tsa.plot_acf(a, lags=40)
#sm.graphics.tsa.plot_pacf(a, lags=40)
# print(np.corrcoef(1/c[:,1],1/c[:,2]))
# print(np.corrcoef(a1*1/c[:,1],a2*1/c[:,2]))
# print(np.corrcoef(b1*1/c[:,1],b2*1/c[:,2]))
#sns.histplot(a1*b*c[:,1])
#plt.show()
#plt.clf()

#A=np.random.gamma(shape=2,scale=1,size=1000000)
# print([np.mean(a),np.mean(b),np.mean(c[:,2])])


print('Inicia generacion de datos')
generar_datos=True
if generar_datos:
  np.random.seed(77)
  start = timer()
  # Prealoca las matrices para almacenar resultados
  X_train_list = []
  y_train_list = []
  
  y_train_phi_vector = np.random.uniform(0,1,size=K)
  y_train_sigma_vector= np.random.uniform(0,3,size=K)#np.random.gamma(shape=2,scale=1,size=K)
  y_train_beta1_vector = np.random.uniform(0,1,size=K)
  y_train_beta3_vector = np.random.gamma(shape=1/3,scale=100,size=K)
  y_train_rho_vector =  np.random.uniform(0,2*np.max(dist_mat),size=K)
  
  

  
  for itera in range(K):
    
    print_progress(itera, checkpoints)
    
    y_train_phi_auxiliar = y_train_phi_vector[itera]
    y_train_sigma_auxiliar = y_train_sigma_vector[itera]
    y_train_beta1_auxiliar = y_train_beta1_vector[itera]
    y_train_gamma_auxiliar = previa_covariables(len(cov[0,:]))
    y_train_beta3_auxiliar = y_train_beta3_vector[itera]
    y_train_rho_auxiliar   = y_train_rho_vector[itera]
    y_train_beta3_auxiliar = np.max([y_train_beta3_auxiliar, 2])  # Restringimos la preva
    y_train_phi_auxiliar = np.min([y_train_phi_auxiliar, 0.95])  # Restringimos la previa para evitar indefiniciones
    y_train_phi_auxiliar = np.max([y_train_phi_auxiliar, 0.05])  # Restringimos la previa para evitar indefiniciones
    for j in range(J):
      y_train_auxiliar = [np.log(y_train_phi_auxiliar/(2-y_train_phi_auxiliar)),
                        np.log(y_train_sigma_auxiliar/(4-y_train_sigma_auxiliar)),
                        np.log(y_train_beta1_auxiliar/(2-y_train_beta1_auxiliar)),
                      -np.log(y_train_beta3_auxiliar-1),
                      np.log(y_train_rho_auxiliar/(rho_upper_range-y_train_rho_auxiliar)),
                      y_train_gamma_auxiliar[0],y_train_gamma_auxiliar[1],y_train_gamma_auxiliar[2],y_train_gamma_auxiliar[3]]
      X_train_auxiliar = np.zeros((nsites, m))
      
      X3_auxiliar_completo = simular_X3(m,y_train_rho_auxiliar,y_train_beta3_auxiliar,nsites,dist_mat) 
      for sitio in range(nsites):
        X1_auxiliar=simular_X1(m,y_train_beta1_auxiliar)
        X2_auxiliar=simular_logAR1(m,y_train_phi_auxiliar,y_train_sigma_auxiliar)
        X3_auxiliar=X3_auxiliar_completo[:,sitio]
        covariables_auxiliar = calculo_covariable(cov[sitio,:],y_train_gamma_auxiliar)
        X_train_auxiliar[sitio] = covariables_auxiliar*X1_auxiliar*X2_auxiliar*X3_auxiliar
      # Hacemos reshape
      X_train_auxiliar = X_train_auxiliar.reshape(1, nsites, m).transpose(0, 2, 1)

      #print(X_train_auxiliar[:,1])
      no_hay_problemas = True
      for lugares in range(nsites):
        if (np.round(X_train_auxiliar[:, :, lugares][0],3)==0).all() or (X_train_auxiliar[:, :, lugares][0]==np.inf).any() or (X_train_auxiliar[:, :, lugares][0]==-np.inf).any():
          no_hay_problemas = False 
          
        
      # Acumulamos
      if no_hay_problemas:
        #print('Cuidado hay un vector de ceros o infinitos, phi: ' + str(y_train_phi_auxiliar) + ' rho: ' + str(y_train_rho_auxiliar))
        X_train_list.append(X_train_auxiliar)
        y_train_list.append(y_train_auxiliar)
      
      
      
  # Convertir listas a arrays
  X_train = np.concatenate(X_train_list, axis=0)
  y_train = np.array(y_train_list)
  end = timer()
  tiempo_simulacion = end - start
  print(tiempo_simulacion)
  
  np.save('X_train-K10k-m100-J100-X1X2X3-log-matriz-covariables-uniforme-todoescalado.npy', X_train) # save
  np.save('y_train-K10k-m100-J100-X1X2X3-log-matriz-covariables-uniforme-todoescalado.npy', y_train) # save
  print('Termina generacion de datos')
else:
  #X_train = np.load('X_train-K10k-m200-J100-X1X2X3-log-matriz-covariables-uniforme-todoescalado.npy',)
  #y_train = np.load('y_train-K10k-m200-J100-X1X2X3-log-matriz-covariables-uniforme-todoescalado.npy')
  print('Termina carga de datos')





print(len(y_train))
print(pd.DataFrame(y_train).describe())

# Mejores arquitecturas hasta el momento.


###
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001,patience=3,restore_best_weights=True)
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',min_delta=0.01,patience=2,restore_best_weights=True)
###





print('Inicia entrenamiento')
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
timesteps, features = m, nsites
inputs = tf.keras.layers.Input((timesteps, features))
x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)


output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# Covariables
output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)

model = tf.keras.Model(inputs, [output1,output2,output3,output4,output5,output6,output7,output8,output9]) #OJO output4
model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
start = timer()
model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7],y_train[:,8]
],epochs=25,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
end = timer()
tiempo_entrenamiento = end - start

print(tiempo_entrenamiento)
model.save('K10k-m100-J100-lstm-X1X2X3-log-matriz-covariables-100(tanh)*5-100(tanh)-100(relu)-batch-128-uniforme-todoescalado')
print('Se guardo el modelo.')


#######################




print('Inicia entrenamiento')
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
timesteps, features = m, nsites
inputs = tf.keras.layers.Input((timesteps, features))
x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)


output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# Covariables
output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)

model = tf.keras.Model(inputs, [output1,output2,output3,output4,output5,output6,output7,output8,output9]) #OJO output4
model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
start = timer()
model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7],y_train[:,8]
],epochs=25,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
end = timer()
tiempo_entrenamiento = end - start

print(tiempo_entrenamiento)
model.save('K10k-m100-J100-lstm-X1X2X3-log-matriz-covariables-100(tanh)*10-100(tanh)-100(relu)-batch-128-uniforme-todoescalado')
print('Se guardo el modelo.')


#######################




#######################




print('Inicia entrenamiento')
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
timesteps, features = m, nsites
inputs = tf.keras.layers.Input((timesteps, features))
x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)


output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# Covariables
output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)

model = tf.keras.Model(inputs, [output1,output2,output3,output4,output5,output6,output7,output8,output9]) #OJO output4
model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
start = timer()
model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7],y_train[:,8]
],epochs=25,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
end = timer()
tiempo_entrenamiento = end - start

print(tiempo_entrenamiento)
model.save('K10k-m100-J100-lstm-X1X2X3-log-matriz-covariables-100(tanh)*5-100(tanh)-100(relu)*5-batch-128-uniforme-todoescalado')
print('Se guardo el modelo.')


#######################



#######################




print('Inicia entrenamiento')
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
timesteps, features = m, nsites
inputs = tf.keras.layers.Input((timesteps, features))
x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(100, activation = "relu")(x)


output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# Covariables
output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)

model = tf.keras.Model(inputs, [output1,output2,output3,output4,output5,output6,output7,output8,output9]) #OJO output4
model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
start = timer()
model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7],y_train[:,8]
],epochs=25,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
end = timer()
tiempo_entrenamiento = end - start

print(tiempo_entrenamiento)
model.save('K10k-m100-J100-lstm-X1X2X3-log-matriz-covariables-100(tanh)*10-100(tanh)-100(relu)*5-batch-128-uniforme-todoescalado')
print('Se guardo el modelo.')


#######################





print('Inicia entrenamiento')
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
timesteps, features = m, nsites
inputs = tf.keras.layers.Input((timesteps, features))
x = tf.keras.layers.LSTM(1000, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(1000, return_sequences=False)(x)
x = tf.keras.layers.Dense(1000, activation = "relu")(x)


output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# Covariables
output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)


model = tf.keras.Model(inputs, [output1,output2,output3,output4,output5,output6,output7,output8,output9]) #OJO output5
model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
start = timer()
model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7],y_train[:,8]
],epochs=50,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
end = timer()
tiempo_entrenamiento = end - start

print(tiempo_entrenamiento)
model.save('K10k-m100-J100-lstm-X1X2X3-log-matriz-covariables-1000(tanh)-1000(tanh)-1000(relu)-batch-128-uniforme-todoescalado')

print('Se guardo el modelo.')











# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(1000, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(1000, return_sequences=False)(x)
# x = tf.keras.layers.Dense(1000, activation = "relu")(x)
# 
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# 
# model = tf.keras.Model(inputs, [output1,output2,output3,output4,output6,output7,output8,output9]) #OJO output5
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7]#,y_train[:,8]
# ],epochs=20,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K20k-m200-J100-lstm-X1X2X3-log-matriz-covariables-beta1fijo-1000(tanh)-1000(tanh)-1000(relu)-batch-128-uniforme-beta3fijo')
# 
# print('Se guardo el modelo.')
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# model = tf.keras.Model(inputs, [output1,output2,output3,output4,output6,output7,output8,output9]) #OJO output5
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6],y_train[:,7]#,y_train[:,8]
# ],epochs=50,batch_size=128,shuffle=True,use_multiprocessing=True,workers=workers_clus,callbacks=[early_stopping],validation_split=0.2)
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K20k-m100-J100-lstm-X1X2X3-log-matriz-covariables--beta1fijo-100(tanh)*5-100(tanh)-100(relu)*5-batch-128-uniforme-beta3fijo')
# print('Se guardo el modelo.')
# 
# 
# 










# Ya intentados con X1X2 intentados
# 
# 
# 
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=64,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-64-uniforme')
# 
# print('Se guardo el modelo.')
# 
# 
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=256,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-256-uniforme')
# 
# print('Se guardo el modelo.')
# 
# #############################################################################################################################
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=128,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-batch-128-uniforme')
# 
# print('Se guardo el modelo.')
# 
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=64,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-batch-64-uniforme')
# 
# print('Se guardo el modelo.')
# 
# 
# 
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# 
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=256,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-batch-256-uniforme')
# 
# print('Se guardo el modelo.')





### Nuevas ideas


# Muy pesado

# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(10000, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(10000, return_sequences=False)(x)
# x = tf.keras.layers.Dense(10000, activation = "relu")(x)
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=128,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-10000(tanh)-10000(tanh)-10000(relu)-batch-128-uniforme')




# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=100,batch_size=128,shuffle=True,use_multiprocessing=True,workers=13,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)*5-100(relu)*5-batch-128-uniforme')



# Da mal

# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=False)(inputs)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.001,)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=128,shuffle=True,use_multiprocessing=True,workers=8,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(relu)-batch-128-uniforme')


# Pesadisimo

# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.01)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=50,batch_size=100000,shuffle=True,use_multiprocessing=True,workers=8,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-100k-uniforme')

# No mejora tanto

# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True,use_bias=False)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False,use_bias=False)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.01)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=20,batch_size=128,shuffle=True,use_multiprocessing=True,workers=8,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-128-uniforme-nobias')

# No mejora tanto


# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True,dropout=0.1)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False,dropout=0.1)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.01)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=20,batch_size=128,shuffle=True,use_multiprocessing=True,workers=8,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-128-uniforme-dropout0.1')
# 
# No mejora tanto
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True,go_backwards=True)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False,go_backwards=True)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.01)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=20,batch_size=128,shuffle=True,use_multiprocessing=True,workers=8,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-128-uniforme-backwards')
#
# No mejora tanto
# 
# print('Inicia entrenamiento')
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# timesteps, features = m, nsites
# inputs = tf.keras.layers.Input((timesteps, features))
# x = tf.keras.layers.LSTM(100, return_sequences=True,recurrent_dropout=0.1)(inputs)
# x = tf.keras.layers.LSTM(100, return_sequences=False,recurrent_dropout=0.1)(x)
# x = tf.keras.layers.Dense(100, activation = "relu")(x)
# 
# output1 =  tf.keras.layers.Dense(1, activation = "linear", name='output1')(x)
# output2 =  tf.keras.layers.Dense(1, activation = "linear", name='output2')(x)
# output3 = tf.keras.layers.Dense(1, activation = "linear", name='output3')(x)
# #output4 = tf.keras.layers.Dense(1, activation = "linear", name='output4')(x)
# #output5 = tf.keras.layers.Dense(1, activation = "linear", name='output5')(x)
# # Covariables
# output6 = tf.keras.layers.Dense(1, activation = "linear", name='output6')(x)
# output7 = tf.keras.layers.Dense(1, activation = "linear", name='output7')(x)
# output8 = tf.keras.layers.Dense(1, activation = "linear", name='output8')(x)
# output9 = tf.keras.layers.Dense(1, activation = "linear", name='output9')(x)
# 
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',min_delta=0.01)
# model = tf.keras.Model(inputs, [output1,output2,output3,output6,output7,output8,output9]) #OJO output4
# model.compile(loss='mse', metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# start = timer()
# model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]],epochs=20,batch_size=128,shuffle=True,use_multiprocessing=True,workers=8,callbacks=[early_stopping])
# end = timer()
# tiempo_entrenamiento = end - start
# 
# print(tiempo_entrenamiento)
# model.save('K10k-m100-J100-lstm-X1X2-matriz-covariables-varios-100(tanh)-100(tanh)-100(relu)-batch-128-uniforme-recurrent0.1')






