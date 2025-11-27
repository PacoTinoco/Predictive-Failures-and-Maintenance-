
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuración de conexión a PostgreSQL ---
DB_HOST = os.getenv("DB_HOST", "your-db-host")
DB_NAME = os.getenv("DB_NAME", "your-db-name")
DB_USER = os.getenv("DB_USER", "your-db-user")
DB_PASS = os.getenv("DB_PASS", "your-db-password")

# --- Ruta donde se guardará el modelo en Render ---
MODEL_PATH = "model.pkl"

def get_data_from_postgres():
    """Extrae datos históricos y nuevos desde PostgreSQL."""
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    query = """
    SELECT maquina, velocidad, temperatura, tipo_capsula, tipo_falla
    FROM eventos_falla;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    """Preprocesa datos (codificación simple para demo)."""
    df = df.dropna()
    X = pd.get_dummies(df.drop("tipo_falla", axis=1))
    y = df["tipo_falla"]
    return X, y

def retrain_model():
    """Reentrena el modelo y lo guarda."""
    print("🔄 Iniciando reentrenamiento...")
    df = get_data_from_postgres()
    X, y = preprocess_data(df)

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo (RandomForest como ejemplo)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación rápida
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Modelo reentrenado. Accuracy: {acc:.2f}")

    # Guardar modelo en Render
    joblib.dump(model, MODEL_PATH)
    print(f"📦 Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    retrain_model()
