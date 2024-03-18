import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('defunciones.csv')

# Resumen de las variables numéricas
print("Resumen de las variables numéricas:")
print(data.describe())

# Obtener el número de columnas numéricas
num_cols = len(data.select_dtypes(include=[np.number]).columns)

# Calcular el número de filas y columnas necesarias para acomodar todos los histogramas
num_rows = (num_cols + 2) // 3  # Asegura que se ajuste al menos 2 columnas por fila
num_cols = min(num_cols, 3)

# Visualización de histogramas para las variables numéricas
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

# Iterar sobre las columnas numéricas y trazar histogramas
for i, column in enumerate(data.select_dtypes(include=[np.number]).columns):
    data[column].hist(ax=axes[i], bins=30)
    axes[i].set_title(column)

# Eliminar ejes no utilizados
for j in range(i + 1, len(axes)):
    axes[j].remove()

plt.tight_layout()
plt.show()

# Tablas de frecuencia para las variables categóricas
print("\nTablas de frecuencia para las variables categóricas:")
for column in data.select_dtypes(include=['object']).columns:
    print("\nVariable:", column)
    print(data[column].value_counts())


# Extraer los datos de la columna "edad"
Mes = data['Mesreg']
Año = data['Añoreg']
Dia = data['Diaocu']
Mes2 = data['Mesocu']
año2 = data['Añoocu']
Edad = data['Edadif']


# Generar una distribución normal a partir de los datos de edades
media_mes = Mes.mean()
desviacion_mes = Mes.std()
normal = np.random.normal(loc=media_mes, scale=desviacion_mes, size=len(Mes))

# Graficar un histograma para visualizar la distribución
plt.hist(normal, bins=30, color='lightblue')
plt.title('Histograma de una distribución normal basada en el mes de registro')
plt.xlabel('Mes')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Generar una distribución normal a partir de los datos de edades
media_año = Año.mean()
desviacion_año = Año.std()
normal2 = np.random.normal(loc=media_año, scale=desviacion_año, size=len(Año))

# Graficar un histograma para visualizar la distribución
plt.hist(normal2, bins=30, color='lightblue')
plt.title('Histograma de una distribución normal basada en el año de registro')
plt.xlabel('Año')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Generar una distribución normal a partir de los datos de edades
media_dia = Dia.mean()
desviacion_dia = Dia.std()
normal3 = np.random.normal(loc=media_dia, scale=desviacion_dia, size=len(Dia))

# Graficar un histograma para visualizar la distribución
plt.hist(normal3, bins=30, color='lightblue')
plt.title('Histograma de una distribución normal basada en el día de ocurrencia')
plt.xlabel('Día')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Generar una distribución normal a partir de los datos de edades
media_mes2 = Mes2.mean()
desviacion_mes2 = Mes2.std()
normal4 = np.random.normal(loc=media_mes2, scale=desviacion_mes2, size=len(Mes2))

# Graficar un histograma para visualizar la distribución
plt.hist(normal4, bins=30, color='lightblue')
plt.title('Histograma de una distribución normal basada en el mes de ocurrencia')
plt.xlabel('Mes')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Generar una distribución normal a partir de los datos de edades
media_año2 = año2.mean()
desviacion_año2 = año2.std()
normal5 = np.random.normal(loc=media_año2, scale=desviacion_año2, size=len(año2))

# Graficar un histograma para visualizar la distribución
plt.hist(normal5, bins=30, color='lightblue')
plt.title('Histograma de una distribución normal basada en el año de ocurrencia')
plt.xlabel('Año')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Generar una distribución normal a partir de los datos de edades
media_edades = Edad.mean()
desviacion_edades = Edad.std()
datos_normal = np.random.normal(loc=media_edades, scale=desviacion_edades, size=len(Edad))

# Graficar un histograma para visualizar la distribución
plt.hist(datos_normal, bins=30, color='lightblue')
plt.title('Histograma de una distribución normal basada en la edad del difunto')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()