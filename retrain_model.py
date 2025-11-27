
# retrain_model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import psycopg2

from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_data_from_postgres():
    """Extrae datos de la tabla eventos_falla para entrenamiento."""
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("Variables de entorno de BD no configuradas.")

    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
    query = """
        SELECT maquina, velocidad, temperatura, tipo_capsula, tipo_falla
        FROM eventos_falla
        WHERE tipo_falla IS NOT NULL;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess(df: pd.DataFrame):
    """Preprocesamiento mínimo: eliminar NAs, one-hot de categóricas."""
    df = df.dropna()
    y = df["tipo_falla"].astype(str)
    X = pd.get_dummies(df.drop(columns=["tipo_falla"]))
    return X, y

def retrain():
    print("🔄 Iniciando reentrenamiento…")
    df = get_data_from_postgres()
    if df.empty:
        print("⚠️ No hay datos en BD para entrenar.")
        return

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Modelo reentrenado. Accuracy: {acc:.3f}")

    # Guardar modelo y meta (columnas esperadas)
    joblib.dump(model, MODEL_PATH)
    print(f"📦 Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    retrain()
