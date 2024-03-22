import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Solicitar o caminho do arquivo de dados
data_path = input("Por favor, insira o nome completo do arquivo CSV de dados: ")

# Leitura e Exploração de Dados
print("\nExplorando os dados:")
data = pd.read_csv(data_path)
print(data.head())
print("\nEstatísticas descritivas:")
print(data.describe())
print("\nVerificando valores nulos:")
print(data.isnull().sum())

# Pré-processamento de Dados
print("\nPré-processamento de dados:")

# Normalizando as características
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('style', axis=1))
y = data['style']

# Divisão do Conjunto de Treinamento e Teste
print("\nDivisão do conjunto de treinamento e teste:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementação do LinearSVC
print("\nImplementação do LinearSVC:")
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Avaliação do Modelo LinearSVC
print("\nAvaliação do Modelo LinearSVC:")
y_pred = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred)
print("Acurácia do LinearSVC:", accuracy_svm)

# Comparação com DummyClassifier
print("\nComparação com DummyClassifier:")
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)
y_pred_dummy = dummy_model.predict(X_test)
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
print("Acurácia do DummyClassifier:", accuracy_dummy)

# Análise e Discussão
print("\nAnálise e Discussão:")
print("Diferença de acurácia entre LinearSVC e DummyClassifier:", accuracy_svm - accuracy_dummy)