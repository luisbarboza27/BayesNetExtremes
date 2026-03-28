
inicio <- Sys.time()
cat("Hora inicio:", format(inicio, "%H:%M:%S"), "\n")

paquetes <- c("tidyr","dplyr",'readr','future.apply','matrixStats', "data.table", "lubridate")

for (p in paquetes) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}



# library(raster)
# library(geodata)
# library(sf)
# library(gridExtra)
# library(SpatialExtremes)




simular_logAR1 <- function(n, phi, sigma) {
  observaciones <- numeric(n)
  ce <- -(sigma^2) / (2 * (1 - phi^2))
  observaciones[1] <- exp(rnorm(1, mean = ce, sd = sigma / sqrt(1 - phi^2)))
  
  for (t in 2:n) {
    observaciones[t] <- exp((1 - phi) * ce + phi * log(observaciones[t - 1]) + rnorm(1, mean = 0, sd = sigma))
  }
  
  return(observaciones)
}

simular_X1 <- function(n, beta1) {
  X1 <- (rexp(n, rate = 1) ^ beta1) / gamma(1 + beta1)
  return(X1)
}

simular_X3 <- function(n, rho, beta3, nsites, dist_mat) {
  # Crear la matriz de covarianzas Sigma
  Sigma <- exp(-dist_mat / rho)
  
  # Simulación de vectores gaussianos multivariados independientes
  Gauss <- MASS::mvrnorm(n, mu = rep(0, nsites), Sigma = Sigma)
  
  # Transformación según la distribución Gamma
  X3 <- qgamma(pnorm(Gauss), shape = beta3, scale = 1) / (beta3 - 1)
  
  return(1/X3)
}

calculo_covariable <- function(fila_locacion, gamma_cov) {
  suma <- 0
  for (z in 1:length(gamma_cov)) {
    suma <- suma + gamma_cov[z] * fila_locacion[z]
  }
  return(exp(suma))
}





set.seed(1)

# Leer el archivo CSV
datos_guanacaste <- read.csv('datosPrecGuanacaste.csv')
covariables_guanacaste <- read.csv('covariables_guanacaste.csv')
# Redondear latitud y longitud
datos_guanacaste$lon <- round(datos_guanacaste$lon, 3)
datos_guanacaste$lat <- round(datos_guanacaste$lat, 3)

# Ordenar por fecha
datos_guanacaste <- datos_guanacaste[order(datos_guanacaste$date), ]
datos_guanacaste <- datos_guanacaste %>% 
  filter(date>=ymd(20200901)) # CAmbio




# Agrupar por latitud y longitud
loc <- covariables_guanacaste
loc <- loc[order(loc$lat, loc$lon, decreasing = c(T, F)), ]
loc <- loc[, c('lon', 'lat', 'alt')]

# Filtrar datos de entrenamiento
locaciones_completas <- loc[loc$lon > -85.7 & loc$lon < -85.2 & loc$lat > 10.3 & loc$lat < 10.8, ]
nsites <- nrow(locaciones_completas)



# Filtrar datos de prueba (test)
filtradora <- locaciones_completas %>% 
  mutate(lonlat = paste0(lon,lat))

locaciones_test <- covariables_guanacaste %>% 
  filter(!(paste0(lon,lat) %in% filtradora$lonlat))


# Concatenar los datos de prueba
locaciones_test <- locaciones_test[order(locaciones_test$lat, locaciones_test$lon, decreasing = c(T, F)), ]

# Concatenar datos de entrenamiento y prueba
locaciones_train_test <- covariables_guanacaste
locaciones_train_test <- locaciones_train_test[order(locaciones_train_test$lat, locaciones_train_test$lon, decreasing = c(T, F)), ]

# Renombrar columnas
locaciones_train_test$lat_x <- locaciones_train_test$lat
locaciones_train_test$lon_x <- locaciones_train_test$lon
locaciones_train_test <- locaciones_train_test %>% dplyr::select(-lat,-lon)



# Hacer merge con locaciones_test
locaciones_train_test <- locaciones_train_test  %>% mutate(lonlat=paste0(lat_x,lon_x))%>% left_join( locaciones_test %>% mutate(lonlat=paste0(lat,lon)), "lonlat")

# Agregar la columna 'tipo' para indicar si es 'train' o 'test'
locaciones_train_test$tipo <- ifelse(is.na(locaciones_train_test$lon), 'train', 'test')

# Seleccionar columnas relevantes
locaciones_train_test <- locaciones_train_test[, c('lon_x', 'lat_x','alt.x', 'tipo')]

# Renombrar columnas
colnames(locaciones_train_test)[1:3] <- c('lon', 'lat','alt')

# Obtener el número de sitios en el conjunto de datos final
nsites_train_test <- nrow(locaciones_train_test)


# Definir las covariables espaciales
Z1 <- locaciones_completas$lon  # primera covariable espacial
Z2 <- locaciones_completas$lat  # segunda covariable espacial
Z12 <- locaciones_completas$lon^2  # primera covariable espacial al cuadrado
Z22 <- locaciones_completas$lat^2  # segunda covariable espacial al cuadrado
Z3 <- locaciones_completas$alt  # tercera covariable (altitud)
Z32 <- locaciones_completas$alt^2  # altitud al cuadrado

# Definir las covariables espaciales para el conjunto de entrenamiento y prueba
Z1_train_test <- locaciones_train_test$lon  # primera covariable espacial
Z2_train_test <- locaciones_train_test$lat  # segunda covariable espacial
Z12_train_test <- locaciones_train_test$lon^2  # primera covariable espacial al cuadrado
Z22_train_test <- locaciones_train_test$lat^2  # segunda covariable espacial al cuadrado
Z3_train_test <- locaciones_train_test$alt  # tercera covariable (altitud)
Z32_train_test <- locaciones_train_test$alt^2  # altitud al cuadrado

# Función para normalización Min-Max
normalize_min_max <- function(x,minz,maxz) {
  return((x -minz) / (maxz- minz))
}

# Normalizar las covariables
a <- min(Z1)
b <- max(Z1)
Z1 <- normalize_min_max(Z1,a,b)
Z1_train_test <- normalize_min_max(Z1_train_test,a,b)
a <- min(Z2)
b <- max(Z2)
Z2 <- normalize_min_max(Z2,a,b)
Z2_train_test <- normalize_min_max(Z2_train_test,a,b)

a <- min(Z12)
b <- max(Z12)

Z12 <- normalize_min_max(Z12,a,b)
Z12_train_test <- normalize_min_max(Z12_train_test,a,b)

a <- min(Z22)
b <- max(Z22)


Z22 <- normalize_min_max(Z22,a,b)
Z22_train_test <- normalize_min_max(Z22_train_test,a,b)

a <- min(Z3)
b <- max(Z3)

Z3 <- normalize_min_max(Z3,a,b)
Z3_train_test <- normalize_min_max(Z3_train_test,a,b)

a <- min(Z32)
b <- max(Z32)

Z32 <- normalize_min_max(Z32,a,b)
Z32_train_test <- normalize_min_max(Z32_train_test,a,b)

# Calcular la matriz de distancia entre los sitios
dist_mat <- as.matrix(dist(locaciones_completas[, c("lon", "lat")]))  # matriz de distancia (nsites x nsites)
dist_mat_train_test <- as.matrix(dist(locaciones_train_test[, c("lon", "lat")]))  # matriz de distancia para train test

# Mostrar el rango de la distancia para verificar el valor máximo
rho_upper_range <- 2 * max(dist_mat)

# Crear la matriz de covariables para el conjunto de datos de entrenamiento
cov_original <- cbind(1, Z1, Z2, Z3, Z12, Z22, Z32)  # matriz de diseño (nsites x 7)

# Crear la matriz de covariables para el conjunto de datos de entrenamiento y prueba
cov_original_train_test <- cbind(1, Z1_train_test, Z2_train_test, Z3_train_test, Z12_train_test, Z22_train_test, Z32_train_test)  # matriz de diseño (nsites x 7)

# Calcular el valor de m
m <- nrow(datos_guanacaste) / nrow(loc)

# Mostrar información de las localizaciones
cat(nsites, "locaciones\n")
cat(m, "en tiempo\n")

cat(nsites_train_test, "locaciones en train test\n")
cat(m, "en tiempo\n")

# Filtrar los datos para entrenamiento y prueba
datos_guanacaste_filtrados <- merge(datos_guanacaste, locaciones_completas, by = c("lon", "lat"))
datos_guanacaste_filtrados_train_test <- merge(datos_guanacaste, locaciones_train_test, by = c("lon", "lat"))

#########################################################################
# Inicializar la matriz de covariables para el conjunto de entrenamiento
X_train_guanacaste <- matrix(0, nrow = m, ncol = nsites)

for (sitio in 1:nsites) {
  lon_actual <- locaciones_completas$lon[sitio]
  lat_actual <- locaciones_completas$lat[sitio]
  
  # Filtrar los datos para el sitio actual
  auxi <- datos_guanacaste_filtrados[datos_guanacaste_filtrados$lon == lon_actual & datos_guanacaste_filtrados$lat == lat_actual, ]
  auxi <- auxi[order(auxi$date), ]$chirps  # Ordenar por fecha y extraer los valores de 'chirps'
  
  # Calcular el cuantil 75
  cuantil_75 <- quantile(auxi, 0.75)
  
  # Asignar el valor a la matriz X_train_guanacaste
  X_train_guanacaste[,sitio] <- ifelse(auxi < cuantil_75, cuantil_75, auxi)
}



########################################################################


# Inicializar la matriz de covariables para el conjunto de entrenamiento y prueba
X_train_guanacaste_train_test <- matrix(0, nrow = m, ncol = nsites_train_test)

for (sitio in 1:nsites_train_test) {
  lon_actual <- locaciones_train_test$lon[sitio]
  lat_actual <- locaciones_train_test$lat[sitio]
  
  # Filtrar los datos para el sitio actual
  auxi <- datos_guanacaste_filtrados_train_test[datos_guanacaste_filtrados_train_test$lon == lon_actual & datos_guanacaste_filtrados_train_test$lat == lat_actual, ]
  auxi <- auxi[order(auxi$date), ]$chirps  # Ordenar por fecha y extraer los valores de 'chirps'
  
  # Calcular el cuantil 75
  cuantil_75 <- quantile(auxi, 0.75)
  
  # Asignar el valor a la matriz X_train_guanacaste_train_test
  X_train_guanacaste_train_test[,sitio] <- ifelse(auxi < cuantil_75, cuantil_75, auxi)
}

proceso_estimacion <- function(params_covariables, params_parametros, nsites, m, cov, dist_mat,modelo) {
  # Asignar los parámetros de covariables
  y_train_gamma_auxiliar <- params_covariables
  # Asignar los parámetros de los otros valores
  y_train_phi_auxiliar <- params_parametros[1]
  y_train_sigma_auxiliar <- params_parametros[2]
  
  
  
  
  if (modelo!='D2' & modelo!='D1'){
    y_train_beta3_auxiliar <- params_parametros[3]
    y_train_rho_auxiliar <- params_parametros[4]
    
    # Simular X3 usando la función simular_X3
    X3_auxiliar_completo <-
      simular_X3(m,
                 y_train_rho_auxiliar,
                 y_train_beta3_auxiliar,
                 nsites,
                 dist_mat)
    
    
  }
  
  
  # Inicialización de X_train_auxiliar usando el código anterior
  X_train_auxiliar <- matrix(0, ncol = nsites, nrow = m)
  
  
  if (modelo == 'DY') {
    X2_auxiliar <- simular_X1(m, y_train_phi_auxiliar)
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Simular X2 usando la función simular_logAR1
      X1_auxiliar <-
        simular_X1(m, y_train_sigma_auxiliar)
      
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <-
        X2_auxiliar * X3_auxiliar * covariables_auxiliar * X1_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  
  
  
  if (modelo == 'D8') {
    X1_auxiliar <- simular_X1(m, 0.5)
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Simular X2 usando la función simular_logAR1
      X2_auxiliar <-
        simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
      
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <-
        X2_auxiliar * X3_auxiliar * covariables_auxiliar * X1_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  
  if (modelo == 'D7') {
    X1_auxiliar <- simular_X1(m, 0.5)
    # Simular X2 usando la función simular_logAR1
    X2_auxiliar <-
      simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <-
        X2_auxiliar * X3_auxiliar * covariables_auxiliar * X1_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  if (modelo == 'D6') {
    # Simular X2 usando la función simular_logAR1
    X2_auxiliar <-
      simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      X1_auxiliar <- simular_X1(m, 0.5)
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <-
        X2_auxiliar * X3_auxiliar * covariables_auxiliar * X1_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  if (modelo == 'D5') {
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      X1_auxiliar <- simular_X1(m, 0.5)
      # Simular X2 usando la función simular_logAR1
      X2_auxiliar <-
        simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <-
        X2_auxiliar * X3_auxiliar * covariables_auxiliar * X1_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  if (modelo == 'D4') {
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Simular X2 usando la función simular_logAR1
      X2_auxiliar <-
        simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <- X2_auxiliar * X3_auxiliar * covariables_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  if (modelo == 'D3') {
    # Simular X2 usando la función simular_logAR1
    X2_auxiliar <-
      simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Obtener el valor correspondiente de X3 para el sitio
      X3_auxiliar <- X3_auxiliar_completo[, sitio]
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <- X2_auxiliar * X3_auxiliar * covariables_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  
  if (modelo == 'D2') {
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Simular X2 usando la función simular_logAR1
      X2_auxiliar <-
        simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
      # Obtener el valor correspondiente de X3 para el sitio
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <- X2_auxiliar * covariables_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  if (modelo == 'D1') {
    # Simular X2 usando la función simular_logAR1
    X2_auxiliar <-
      simular_logAR1(m, y_train_phi_auxiliar, y_train_sigma_auxiliar)
    
    # Bucle para calcular las covariables y valores para cada sitio
    for (sitio in 1:nsites) {
      # Obtener el valor correspondiente de X3 para el sitio
      
      # Calcular la covariable auxiliar
      covariables_auxiliar <-
        calculo_covariable(cov[sitio,], y_train_gamma_auxiliar)
      
      # Calcular los valores auxiliares
      auxi <- X2_auxiliar * covariables_auxiliar
      
      # Calcular el cuantil 75
      cuantil_75 <- quantile(auxi, 0.75)
      
      # Asignar el valor mínimo entre auxi y el cuantil 75
      X_train_auxiliar[, sitio] <-
        ifelse(auxi < cuantil_75, cuantil_75, auxi)
    }
    
    # Retornar la matriz final
    return(X_train_auxiliar)
  }
  
  
}


df_X_train_guanacaste_train_test <- data.frame(t=1:m,X_train_guanacaste_train_test) %>% 
  pivot_longer(X1:X83, names_to = "variable", values_to = "valor") 








# Profe
##################################################
# Aqui se define cuales son los modelos a ejecutar, depende de las capacidades del servidor debería escogerse cuales correr.

# Todos los modelos son:
#combinations <- expand.grid(D = c("1","2","3","4","5","6","7","8","Y","Z"), M = 1:7)
# Se corre en for porque el uso de memoria puede ser muy alto.
##################################################

# Crear carpeta de salida si no existe
dir.create("calculo_WIS", showWarnings = FALSE)
combinations <- expand.grid(D =  c("1","2","3","4","5","6","7","8","Y"), M = 1:7)
library(parallel)
library(readr)
library(dplyr)
library(tidyr)

# Crear carpeta de salida si no existe
dir.create("calculo_WIS", showWarnings = FALSE)

combinations <- expand.grid(D =  c("1","2","3","4","5","6","7","8","Y"), M = 1:7)

# Número de núcleos
n_cores <- 4
cl <- makeCluster(n_cores, outfile = "")

# Exportar objetos necesarios a los workers
clusterExport(cl, varlist = c(
  "combinations",
  "cov_original_train_test",
  "nsites_train_test",
  "m",
  "dist_mat_train_test",
  "df_X_train_guanacaste_train_test",
  "proceso_estimacion",
  "simular_logAR1",
  "simular_X1",
  "simular_X3",
  "calculo_covariable"
))

clusterEvalQ(cl, {
  library(readr)
  library(dplyr)
  library(tidyr)
  library(matrixStats)
  library(lubridate)
})

# Función que ejecuta UN modelo
procesar_modelo <- function(i) {
  
  D <- combinations$D[i]
  M <- combinations$M[i]
  
  modelo_parametros <- paste0('D', D)
  modelo_covariables <- paste0('M', M)
  modelo_completo <- paste0(modelo_parametros,'-', modelo_covariables)
  
  log_file <- paste0("log_", modelo_completo,'-test', ".txt")
  
  log_msg <- function(txt) {
    cat(paste0(Sys.time(), " | ", txt, "\n"),
        file = log_file, append = TRUE)
  }
  
  log_msg(paste("Inicio modelo", modelo_completo))
  
  modelo_ubicacion <- sprintf(
    "Traceplot_aplicacion_phi_final/trace_covariables_D%s_aplicacion_M%s.csv", 
    D, M
  )
  
  log_msg(paste("Leyendo archivo", modelo_ubicacion))
  
  trace_modelo <- read_csv(modelo_ubicacion, show_col_types = FALSE)
  trace_modelo <- trace_modelo[, 2:ncol(trace_modelo)]
  
  id_covariables <- which(startsWith(names(trace_modelo), 'posterior_g'))
  
  if (D %in% c(1,2)){
    id_parametros <- c(1,2)
  } else {
    id_parametros <- (length(id_covariables) + 1):ncol(trace_modelo)
  }
  
  cantidad_simulaciones <- nrow(trace_modelo)
  
  log_msg("Inicia simulaciones")
  
  df_simul <- data.frame()
  
  for (muestra in 1:cantidad_simulaciones) {
    
    params_covariables <- unlist(trace_modelo[muestra, id_covariables])
    params_parametros <- unlist(trace_modelo[muestra, id_parametros])
    
    if (length(params_covariables) == 1) {
      covm <- matrix(cov_original_train_test, ncol = 1)
    } else {
      covm <- cov_original_train_test[, 1:length(id_covariables)]
    }
    
    simul <- proceso_estimacion(
      params_covariables, params_parametros,
      nsites_train_test, m, covm,
      dist_mat_train_test, modelo_parametros
    )
    
    df_simul_aux <- data.frame(muestra=muestra,t=1:m,simul)
    df_simul <- rbind(df_simul,df_simul_aux)
  }
  
  log_msg("Inicia percentiles")
  
  df_simul_upper_lower <- df_simul %>% 
    pivot_longer(X1:X83, names_to = "variable", values_to = "valor") %>% 
    group_by(t, variable) %>% 
    summarise(
      percentil = round(seq(0.05, 0.95, by = 0.05),2),
      q = quantile(valor, probs = percentil),
      .groups = "drop"
    )
  
  log_msg("Inicia IS")
  
  vector_upper <- round(seq(0.55,0.95,0.05),2)
  tv <- unique(df_simul_upper_lower$t)
  variablev <- unique(df_simul_upper_lower$variable)
  
  df_IS <- data.frame()
  
  for (tt in tv) {
    for (vv in variablev) {
      for (upper in vector_upper) {
        
        lower <- round(1-upper,2)
        
        lower_upper_t <- df_simul_upper_lower %>% 
          filter(t==tt,variable==vv, percentil %in% c(upper, lower)) %>% 
          arrange(percentil)
        
        mediana_t <- df_simul_upper_lower %>% 
          filter(t==tt,variable==vv, percentil==0.5)
        
        y_t <- df_X_train_guanacaste_train_test %>% 
          filter(t==tt,variable==vv)
        
        IS <- lower_upper_t$q[2]-lower_upper_t$q[1] +
          2/lower*ifelse(lower_upper_t$q[1]>y_t$valor[1],
                         lower_upper_t$q[1]-y_t$valor[1],0) +
          2/lower*ifelse(lower_upper_t$q[2]<y_t$valor[1],
                         y_t$valor[1]-lower_upper_t$q[2],0)
        
        df_IS_aux <- data.frame(
          modelo=modelo_completo,t=tt,variable=vv,
          alpha=lower,
          lower=lower_upper_t$q[1],
          upper=lower_upper_t$q[2],
          y=y_t$valor[1],
          m=mediana_t$q[1],
          IS=IS
        )
        
        df_IS <- rbind(df_IS, df_IS_aux)
      }
    }
  }
  
  archivo <- paste0('WIS_phi_test_', modelo_completo, '.RData')
  ruta <- file.path("calculo_WIS", archivo)
  
  save(df_IS, file = ruta)
  
  log_msg(paste("Fin modelo", modelo_completo))
  
  return(modelo_completo)
}

resultado <- tryCatch({
  parLapply(cl, 1:nrow(combinations), procesar_modelo)
}, error = function(e) {
  print(e)
})


stopCluster(cl)