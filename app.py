

# app.py
import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime, time

# --- Configuración de la página ---
st.set_page_config(page_title="Mantenimiento Predictivo", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model():
    """Carga el modelo si existe; si no, devuelve None."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.title("Mantenimiento Predictivo — Registro y Predicción")

left, right = st.columns([1, 1])

# --- Formulario Izquierdo ---
with left:
    st.subheader("Registro de falla")

    maquina = st.selectbox("Máquina", ["KDF7", "KDF8", "KDF9", "MULFI-10", "KDF11", "KDF17"])
    fecha = st.date_input("Fecha", datetime.now().date())
    hora = st.time_input("Hora", datetime.now().time())
    turno = st.selectbox("Turno", ["1", "2", "3"])
    tipo_capsula = st.selectbox("Tipo de cápsula", ["Gelatina dura", "Gelatina blanda", "Vegana"])
    marca_actual = st.text_input("Marca actual", "Capsugel")
    tipo_falla = st.selectbox("Tipo de falla (catálogo)", [
        "Bloqueo alimentador", "Desalineación cápsula", "Sobrecalentamiento motor",
        "Desgaste rodamiento", "Sellado deficiente", "Otro"
    ])
    descripcion = st.text_area("Descripción de la falla")

    with st.expander("Parámetros extra (CLs y proceso)"):
        velocidad = st.slider("Velocidad (caps/min)", min_value=200, max_value=1800, value=1200, step=10)
        temperatura = st.number_input("Temperatura (°C)", min_value=10.0, max_value=120.0, value=75.0, step=0.5)
        presion = st.number_input("Presión (bar)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

        cl_dosificacion = st.number_input("CL-Dosificación (mg)", min_value=0.0, max_value=1000.0, value=500.0)
        cl_banda_tension = st.number_input("CL-Banda — Tensión (N)", min_value=0.0, max_value=200.0, value=80.0)
        cl_sellado_temp = st.number_input("CL-Sellado — Temperatura (°C)", min_value=20.0, max_value=200.0, value=120.0)
        cl_perforado_offset = st.number_input("CL-Perforado — Offset (mm)", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
        cl_alimentador_gap = st.number_input("CL-Alimentador — Gap (mm)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

        observaciones = st.text_area("Observaciones del operador")

    st.markdown("---")
    st.subheader("Cambio de marca programado")
    hay_cambio = st.checkbox("¿Habrá cambio de marca en el próximo turno?")
    marca_siguiente, hora_cambio, aplicar_impacto = None, None, False
    if hay_cambio:
        marca_siguiente = st.text_input("Marca siguiente", "Qualicaps")
        hora_cambio = st.time_input("Hora aproximada del cambio", value=time(hour=18, minute=0))
        aplicar_impacto = st.checkbox("Aplicar impacto del cambio en la predicción", value=True)

    # Botones
    guardar = st.button("Guardar registro en BD (opcional)")
    predecir = st.button("Predecir próximos escenarios")

# --- (Opcional) Guardado en PostgreSQL ---
def save_event_to_db(payload: dict):
    """Guarda el evento en PostgreSQL. Requiere variables de entorno."""
    import psycopg2
    from psycopg2.extras import execute_values

    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_PORT = os.getenv("DB_PORT", "5432")

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        st.warning("Variables de entorno de BD no configuradas. Saltando guardado.")
        return

    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
    cur = conn.cursor()
    # Crea tabla si no existe
    cur.execute("""
    CREATE TABLE IF NOT EXISTS eventos_falla (
        id SERIAL PRIMARY KEY,
        fecha TIMESTAMP,
        maquina TEXT,
        turno TEXT,
        tipo_capsula TEXT,
        marca_actual TEXT,
        tipo_falla TEXT,
        descripcion TEXT,
        velocidad FLOAT,
        temperatura FLOAT,
        presion FLOAT,
        cl_dosificacion FLOAT,
        cl_banda_tension FLOAT,
        cl_sellado_temp FLOAT,
        cl_perforado_offset FLOAT,
        cl_alimentador_gap FLOAT,
        observaciones TEXT,
        hay_cambio BOOLEAN,
        marca_siguiente TEXT,
        hora_cambio TEXT
    );
    """)
    conn.commit()

    row = (
        datetime.combine(fecha, hora),
        maquina, turno, tipo_capsula, marca_actual, tipo_falla, descripcion,
        velocidad, temperatura, presion, cl_dosificacion, cl_banda_tension,
        cl_sellado_temp, cl_perforado_offset, cl_alimentador_gap,
        observaciones, hay_cambio, marca_siguiente, str(hora_cambio) if hora_cambio else None
    )

    execute_values(cur,
        """INSERT INTO eventos_falla (
            fecha, maquina, turno, tipo_capsula, marca_actual, tipo_falla, descripcion,
            velocidad, temperatura, presion, cl_dosificacion, cl_banda_tension,
            cl_sellado_temp, cl_perforado_offset, cl_alimentador_gap,
            observaciones, hay_cambio, marca_siguiente, hora_cambio
        ) VALUES %s""",
        [row]
    )
    conn.commit()
    cur.close()
    conn.close()
    st.success("Registro guardado en PostgreSQL.")

if guardar:
    payload = {
        "maquina": maquina, "fecha": fecha, "hora": hora, "turno": turno,
        "tipo_capsula": tipo_capsula, "marca_actual": marca_actual,
        "tipo_falla": tipo_falla, "descripcion": descripcion,
        "velocidad": velocidad, "temperatura": temperatura, "presion": presion,
        "cl_dosificacion": cl_dosificacion, "cl_banda_tension": cl_banda_tension,
        "cl_sellado_temp": cl_sellado_temp, "cl_perforado_offset": cl_perforado_offset,
        "cl_alimentador_gap": cl_alimentador_gap,
        "observaciones": observaciones, "hay_cambio": hay_cambio,
        "marca_siguiente": marca_siguiente, "hora_cambio": hora_cambio
    }
    save_event_to_db(payload)

# --- Predicciones (panel derecho) ---
def simulate_predictions(features: dict):
    """Fallback si no hay modelo: simula 3 escenarios."""
    return [
        {
            "tipo": "Desgaste rodamiento",
            "prob": 0.40,
            "eta_h": 6.5,
            "ic": (4.2, 8.1),
            "drivers": ["Velocidad alta", "CL-Alimentador gap ↑", "Cambio de marca en ~2 h"]
        },
        {
            "tipo": "Sobrecalentamiento motor",
            "prob": 0.34,
            "eta_h": 10.0,
            "ic": (7.0, 13.5),
            "drivers": ["Temperatura ↑", "CL-Sellado temp ↑"]
        },
        {
            "tipo": "Bloqueo alimentador",
            "prob": 0.26,
            "eta_h": 15.5,
            "ic": (11.0, 20.0),
            "drivers": ["Tensión de banda baja", "Presión ↓"]
        },
    ]

def predict_with_model(model, features: dict):
    """Ejemplo mínimo de predicción con scikit-learn usando features numéricas."""
    # Convertir features a DataFrame con las columnas esperadas por el modelo
    df = pd.DataFrame([{
        "maquina": features["maquina"],
        "velocidad": features["velocidad"],
        "temperatura": features["temperatura"],
        "tipo_capsula": features["tipo_capsula"]
    }])
    # One-hot simple (debe coincidir con el entrenamiento)
    X = pd.get_dummies(df)
    # Alinear columnas
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        for c in expected_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[expected_cols]
    preds = model.predict_proba(X)[0]  # asumiendo clasificador multiclase
    classes = model.classes_
    # Ordenar top-3
    order = preds.argsort()[::-1][:3]
    escenarios = []
    for idx in order:
        escenarios.append({
            "tipo": classes[idx],
            "prob": float(preds[idx]),
            "eta_h": 8.0,             # placeholder: integrar modelo de supervivencia real
            "ic": (6.0, 12.0),
            "drivers": ["(Explicabilidad pendiente con SHAP)"]
        })
    return escenarios

with right:
    st.subheader("Próximos 3 escenarios de falla")

    if predecir:
        features = {
            "maquina": maquina, "fecha": fecha, "hora": hora, "turno": turno,
            "tipo_capsula": tipo_capsula, "marca_actual": marca_actual,
            "tipo_falla": tipo_falla, "descripcion": descripcion,
            "velocidad": velocidad, "temperatura": temperatura, "presion": presion,
            "cl_dosificacion": cl_dosificacion, "cl_banda_tension": cl_banda_tension,
            "cl_sellado_temp": cl_sellado_temp, "cl_perforado_offset": cl_perforado_offset,
            "cl_alimentador_gap": cl_alimentador_gap,
            "observaciones": observaciones, "hay_cambio": hay_cambio,
            "marca_siguiente": marca_siguiente, "hora_cambio": hora_cambio
        }

        if model is None:
            st.info("No se encontró un modelo entrenado. Mostrando simulación.")
            escenarios = simulate_predictions(features)
        else:
            escenarios = predict_with_model(model, features)

        for i, esc in enumerate(escenarios, start=1):
            st.markdown(f"### Escenario {i}: {esc['tipo']}")
            st.write(f"**Probabilidad:** {esc['prob']*100:.0f}%")
            st.write(f"**ETA:** {esc['eta_h']:.1f} h  —  **IC 95%:** [{esc['ic'][0]:.1f}, {esc['ic'][1]:.1f}] h")
            st.write(f"**Factores más influyentes:** {', '.join(esc['drivers'])}")
            st.markdown("---")
