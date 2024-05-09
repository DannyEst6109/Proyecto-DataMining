# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
df_clean = pd.read_csv("defunciones_clean.csv")

# Tomar una muestra del 20% de los datos
df_sample = df_clean.sample(frac=0.2, random_state=42)

# Convertir variables categóricas en variables dummy utilizando one-hot encoding
X_encoded = pd.get_dummies(df_sample.drop('causa', axis=1), columns=['etnia', 'escolaridad', 'ocupacion', 'lugar'])
y = df_sample['causa']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Escalar las características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear un modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
rf_model.fit(X_train_scaled, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluar el rendimiento del modelo de Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Mostrar el reporte de clasificación del modelo de Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Mostrar la matriz de confusión del modelo de Random Forest
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
