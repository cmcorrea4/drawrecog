import base64
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

MARGEN = 30                 # px libres alrededor del plano (para las etiquetas)
COLOR_TRAZO = "#FF0000"     # el trazo del usuario va en rojo para distinguirlo de la referencia
COLOR_EJES = "#1F3A93"
COLOR_GRILLA = "#D9D9D9"
COLOR_INICIO = (0, 170, 0)  # marca verde del punto de partida


# ---------------------------------------------------------------------------
# Plano cartesiano de referencia
# ---------------------------------------------------------------------------
def paso_grilla(rango):
    """Separación entre líneas de la cuadrícula según el rango del plano."""
    if rango <= 10:
        return 1
    if rango <= 20:
        return 2
    return 5


def a_pixel(x, y, lado, rango):
    """Convierte coordenadas cartesianas a pixeles del canvas."""
    escala = (lado / 2 - MARGEN) / rango
    return lado / 2 + x * escala, lado / 2 - y * escala


def plano_cartesiano(lado, rango):
    """Objetos fabric.js (cuadrícula, ejes y etiquetas) que se cargan en el canvas."""
    paso = paso_grilla(rango)
    centro = lado / 2
    largo = lado - 2 * MARGEN
    objetos = []

    def rect(left, top, width, height, color):
        objetos.append({
            "type": "rect", "left": left, "top": top, "width": width, "height": height,
            "fill": color, "strokeWidth": 0, "selectable": False, "evented": False,
        })

    def texto(left, top, contenido, size=11, color="#444444"):
        objetos.append({
            "type": "text", "left": left, "top": top, "text": contenido,
            "fontSize": size, "fontFamily": "Arial", "fill": color,
            "selectable": False, "evented": False,
        })

    valores = [v for v in range(-rango, rango + 1, paso) if v != 0]

    # Cuadrícula
    for v in valores:
        px, py = a_pixel(v, v, lado, rango)
        rect(px, MARGEN, 1, largo, COLOR_GRILLA)
        rect(MARGEN, py, largo, 1, COLOR_GRILLA)

    # Ejes
    rect(centro - 1, MARGEN, 2, largo, COLOR_EJES)
    rect(MARGEN, centro - 1, largo, 2, COLOR_EJES)

    # Etiquetas numéricas
    for v in valores:
        px, py = a_pixel(v, v, lado, rango)
        texto(px - 3 * len(str(v)), centro + 4, str(v))
        texto(centro + 5, py - 6, str(v))
    texto(centro + 5, centro + 4, "0")
    texto(lado - MARGEN + 8, centro - 8, "X", 14, COLOR_EJES)
    texto(centro - 5, MARGEN - 22, "Y", 14, COLOR_EJES)

    return {"version": "4.4.0", "objects": objetos}


# ---------------------------------------------------------------------------
# Imagen que se envía al LLM
# ---------------------------------------------------------------------------
def inicio_del_trazo(json_data):
    """Pixel donde el usuario empezó a dibujar (primer trazo), o None si no hay trazo."""
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") == "path" and obj.get("path"):
            return obj["path"][0][1], obj["path"][0][2]
    return None


def imagen_para_llm(image_data, inicio, lado):
    """Aplana el canvas sobre fondo blanco y marca en verde el inicio del trazo."""
    img = Image.fromarray(np.array(image_data).astype("uint8"), "RGBA")
    fondo = Image.new("RGBA", img.size, "white")
    img = Image.alpha_composite(fondo, img).convert("RGB")
    f = img.width / lado  # por si el navegador entrega la imagen a otra resolución
    x, y = inicio[0] * f, inicio[1] * f
    r = 7 * f
    ImageDraw.Draw(img).ellipse([x - r, y - r, x + r, y + r], fill=COLOR_INICIO, outline="black")
    return img


# ---------------------------------------------------------------------------
# LLM: de la imagen al vector de puntos
# ---------------------------------------------------------------------------
def construir_prompt(rango, n_puntos):
    paso = paso_grilla(rango)
    return f"""Eres el módulo de visión que planifica la trayectoria de un robot.

La imagen contiene un plano cartesiano de referencia:
- Los ejes X e Y son las líneas azul oscuro; se cruzan en el origen (0, 0), en el centro.
- X crece hacia la derecha y Y crece hacia arriba.
- Ambos ejes van de -{rango} a {rango}. Hay una línea gris de cuadrícula cada {paso} unidad(es), con su valor numérico escrito junto a los ejes.

El trazo ROJO es la trayectoria que dibujó el usuario. El círculo VERDE marca el punto donde empieza.

Tarea: devuelve exactamente {n_puntos} puntos (x, y) que describan la trayectoria.
- Ordénalos desde el inicio (círculo verde) hasta el final del trazo.
- Repártelos de forma aproximadamente uniforme a lo largo del trazo, conservando esquinas y cambios de dirección.
- Lee cada coordenada apoyándote en la cuadrícula y las etiquetas, con un decimal.
- Si hay varios trazos separados, recórrelos uno tras otro empezando por el que tiene el círculo verde.

Responde únicamente con un objeto JSON con esta forma, sin texto adicional:
{{"puntos": [[x1, y1], [x2, y2], ...]}}"""


def pedir_trayectoria(client, modelo, img, rango, n_puntos):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    respuesta = client.chat.completions.create(
        model=modelo,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": construir_prompt(rango, n_puntos)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ],
        }],
    )
    return respuesta.choices[0].message.content


def validar_puntos(contenido, rango):
    """Convierte la respuesta del LLM en una lista [[x, y], ...].

    Los valores se limitan a [-rango, rango] para no enviar al robot
    coordenadas fuera del área de trabajo.
    """
    datos = json.loads(contenido)
    puntos = []
    for p in datos.get("puntos", []):
        if isinstance(p, (list, tuple)) and len(p) == 2:
            x = max(-rango, min(rango, float(p[0])))
            y = max(-rango, min(rango, float(p[1])))
            puntos.append([round(x, 2), round(y, 2)])
    return puntos


def graficar_verificacion(img, puntos, lado, rango):
    """Superpone los puntos del LLM sobre el dibujo original para comprobarlos."""
    limite = (lado / 2) / ((lado / 2 - MARGEN) / rango)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, extent=[-limite, limite, -limite, limite])
    xs, ys = zip(*puntos)
    ax.plot(xs, ys, "o--", color="black", markersize=5, linewidth=1)
    for i, (x, y) in enumerate(puntos):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Puntos del LLM (negro) sobre el trazo original (rojo)")
    return fig


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Tablero Inteligente", layout="wide")
st.title("Tablero Inteligente: trayectoria para robot")

with st.sidebar:
    st.subheader("Acerca de:")
    st.write(
        "Dibuja una trayectoria sobre el plano cartesiano. El modelo la interpreta "
        "y la convierte en un vector de puntos (x, y) para enviar al robot."
    )
    st.divider()
    lado = st.slider("Tamaño del canvas (px)", 400, 800, 600, 50)
    rango = st.select_slider("Rango del plano (±)", options=[5, 10, 20, 50], value=10)
    n_puntos = st.slider("Número de puntos de la trayectoria", 5, 40, 15)
    stroke_width = st.slider("Ancho de línea", 1, 30, 4)
    modelo = st.text_input("Modelo", "gpt-6-luna") #gpt-40-mini

st.subheader("Dibuja la trayectoria (de un solo trazo) y presiona el botón")

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=COLOR_TRAZO,
    background_color="#FFFFFF",
    height=lado,
    width=lado,
    drawing_mode="freedraw",
    initial_drawing=plano_cartesiano(lado, rango),
    key=f"canvas_{lado}_{rango}",  # se reinicia si cambia el tamaño o el rango
)

api_key = st.text_input("Ingresa tu Clave", type="password")
analyze_button = st.button("Generar trayectoria", type="primary")

if analyze_button:
    inicio = inicio_del_trazo(canvas_result.json_data)
    if not api_key:
        st.warning("Por favor ingresa tu API key.")
    elif canvas_result.image_data is None or inicio is None:
        st.warning("Primero dibuja una trayectoria en el plano.")
    else:
        with st.spinner("Analizando el trazo ..."):
            try:
                img = imagen_para_llm(canvas_result.image_data, inicio, lado)
                client = OpenAI(api_key=api_key)
                contenido = pedir_trayectoria(client, modelo, img, rango, n_puntos)
                puntos = validar_puntos(contenido, rango)
                if puntos:
                    st.session_state.trayectoria = {
                        "puntos": puntos, "imagen": img, "lado": lado, "rango": rango,
                    }
                else:
                    st.error("El modelo no devolvió puntos. Intenta de nuevo.")
            except json.JSONDecodeError:
                st.error("La respuesta del modelo no fue un JSON válido. Intenta de nuevo.")
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

# Resultado (se guarda en session_state para que no se pierda al recargar)
if "trayectoria" in st.session_state:
    t = st.session_state.trayectoria
    puntos = t["puntos"]  # <- este es el vector para el robot: [[x, y], ...]
    vector_json = json.dumps(puntos)

    st.subheader("Vector de trayectoria")
    st.code(vector_json, language="json")
    st.download_button("Descargar JSON", vector_json, "trayectoria.json", "application/json")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(graficar_verificacion(t["imagen"], puntos, t["lado"], t["rango"]))
    with col2:
        st.dataframe(pd.DataFrame(puntos, columns=["x", "y"]), use_container_width=True)
