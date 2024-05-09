# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
df_clean = pd.read_csv("defunciones_clean.csv")

# Convertir variables categóricas en variables dummy utilizando one-hot encoding
X_encoded = pd.get_dummies(df_clean.drop('causa', axis=1), columns=['etnia', 'escolaridad', 'ocupacion', 'lugar'])
y = df_clean['causa']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Escalar las características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear un modelo SVM
svm_model = SVC(kernel='rbf', random_state=42)

# Entrenar el modelo
svm_model.fit(X_train_scaled, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test_scaled)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Mostrar el reporte de clasificación
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
