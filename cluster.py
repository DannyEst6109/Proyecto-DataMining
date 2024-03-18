import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Especificar tipos de datos de columnas problemáticas
dtype_dict = {'columna18': str, 'columna28': str, 'columna29': str}

# Cargar el archivo CSV con tipos de datos especificados
data = pd.read_csv('defunciones.csv', dtype=dtype_dict)

# Seleccionar solo las columnas numéricas de interés
variables_numericas = data[['Mesreg', 'Añoreg', 'Diaocu', 'Mesocu', 'Edadif']]

# Copiar los datos para evitar SettingWithCopyWarning
variables_numericas = variables_numericas.copy()

# Manejar valores faltantes si los hay
variables_numericas.dropna(inplace=True)

# Estandarizar las variables
scaler = StandardScaler()
variables_numericas_estandarizadas = scaler.fit_transform(variables_numericas)

# Aplicar el algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(variables_numericas_estandarizadas)

# Verificar los centroides de los clústeres
centroides = scaler.inverse_transform(kmeans.cluster_centers_)
centroides_df = pd.DataFrame(centroides, columns=variables_numericas.columns)
print("Centroides de los clústeres:")
print(centroides_df)

# Agregar los resultados del clustering al DataFrame original
data['Cluster'] = clusters

# Visualización de los clústeres
for cluster in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Añoreg'], cluster_data['Edadif'], label=f'Cluster {cluster}')

plt.scatter(centroides[:, 1], centroides[:, 4], marker='X', color='black', label='Centroides')
plt.xlabel('Añoreg')
plt.ylabel('Edadif')
plt.title('Clustering de Defunciones')
plt.legend()
plt.show()
