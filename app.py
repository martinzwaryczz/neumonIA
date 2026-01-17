# Streamlit para el desarrollo de la interfaz gráfica
import streamlit as st

# PyTorch para la arquitectura IA
from torchvision import transforms
import torch

# Arquitectura creada
from model_architecture import NeumoniaDetector

# PIL para llevar la imagen a tensor luego con el transform correspondiente
from PIL import Image

# SQLITE3 para el manejo del historia y la tabla asociada
import sqlite3

# Pandas para el manejo de la tabla a nivel programa, no BD
import pandas as pd

# Datatime para guardar la fecha y hora de realización de los registros de la tabla
from datetime import datetime


# Icono del titulo asignado con streamlit, analogo al <title> de HTML.
try:
    favicon = Image.open("img/logo.jpg")
    st.set_page_config(page_title="Detector de NeumonIA", page_icon=favicon, layout="centered")
except:
    st.set_page_config(page_title="Detector de NeumonIA", page_icon="🩻", layout="centered")

# Inicializar Base de Datos SQLite, creación de la tabla
def init_db():
    conn = sqlite3.connect('bd/historial_clinico.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analisis 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
         fecha TEXT, 
         hora TEXT, 
         condicion TEXT, 
         confianza REAL)
    ''')
    conn.commit()
    conn.close()

init_db()

#  FUNCIONES DE APOYO Y ESTILOS 

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de estilos en: {file_name}")

local_css("estilos/styles.css")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = NeumoniaDetector(input_channels=3, output_shape=2)
    try:
        model.load_state_dict(torch.load("modelos/modelo_neumonia_v1.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'modelos/modelo_neumonia_v1.pth'")
    return model, device

model, device = load_model()

# BANNER INICIAL
st.markdown("""
    <div class="banner-recto">
        <h1>Analizador de radiografías de pulmones</h1>
        <p>Detección de neumonía con una red neuronal IA hecho con PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

# CARGADOR DE IMAGEN CON LOGICA DE TRANSFORMACIÓN A TENSOR
archivo = st.file_uploader("Subir radiografía", type=["jpg", "jpeg", "png"], key="cargador_radiografias")

if archivo:
    with st.container():
        img = Image.open(archivo).convert('RGB')
        st.image(img, caption="Radiografía para analizar", use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ejecutar_evaluacion = st.button("EVALUAR")

        if ejecutar_evaluacion:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(device)

            with st.spinner('Analizando imagen...'):
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)
            
            condicion = "Neumonía" if pred.item() == 1 else "Normal"
            confianza = round(conf.item() * 100, 2)

            # GUARDAR EN LA BASE DE DATOS
            conn = sqlite3.connect('bd/historial_clinico.db')
            c = conn.cursor()
            ahora = datetime.now()
            c.execute('''
                INSERT INTO analisis (fecha, hora, condicion, confianza) 
                VALUES (?, ?, ?, ?)
            ''', (ahora.strftime("%d/%m/%Y"), ahora.strftime("%H:%M:%S"), condicion, confianza))
            conn.commit()
            conn.close()

            # MOSTRAR RESULTADO 
            if condicion == "Neumonía":
                st.error(f"RESULTADO: {condicion} ({confianza}%)")
            else:
                st.success(f"RESULTADO: {condicion} ({confianza}%)")

# TABLA HISTÓRICA: PARTE MÁS DIFÍCIL, TUVE MUCHOS PROBLEMAS Y ME AYUDO MUCHO GEMMINI AUNQUE TAMBIÉN DIO 1.000 VUELTAS...
st.markdown("---")
st.markdown("### Registro Histórico de Análisis")

# 1. CSS independiente para evitar errores de texto residual
st.markdown("""
    <style>
        .contenedor-ia-final {
            height: 350px;
            overflow-y: auto;
            border: 3px solid black;
            box-shadow: 10px 10px 0px black;
            background-color: white;
            margin-bottom: 20px;
        }
        .tabla-ia-full {
            width: 100% !important;
            border-collapse: collapse;
        }
        .tabla-ia-full th {
            position: sticky;
            top: 0;
            background-color: #0055FF;
            color: white;
            padding: 12px;
            border: 1px solid black;
            z-index: 5;
        }
        .tabla-ia-full td {
            padding: 10px;
            border: 1px solid black;
            text-align: center;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Lógica de datos
try:
    with sqlite3.connect("bd/historial_clinico.db") as conn:
        query = "SELECT id as 'Nº', fecha as Fecha, hora as Hora, condicion as 'Condición', confianza as 'Confianza %' FROM analisis ORDER BY id ASC"
        df_publico = pd.read_sql_query(query, conn)
except:
    df_publico = pd.DataFrame()

if not df_publico.empty:
    # Construcción de filas con generador simple para evitar saturar el markdown
    filas_list = []
    for _, row in df_publico.iterrows():
        color = "red" if row['Condición'] == "Neumonía" else "green"
        conf = f"{float(row['Confianza %']):.2f}"
        fila = f"<tr><td>{row['Nº']}</td><td>{row['Fecha']}</td><td>{row['Hora']}</td><td style='color: {color}; font-weight: bold;'>{row['Condición']}</td><td>{conf}</td></tr>"
        filas_list.append(fila)
    
    html_filas = "".join(filas_list)

    # 3. Renderizado de la estructura HTML
    tabla_html = f"""
        <div class="contenedor-ia-final">
            <table class="tabla-ia-full">
                <thead>
                    <tr>
                        <th>Nº</th><th>Fecha</th><th>Hora</th><th>Condición</th><th>Confianza %</th>
                    </tr>
                </thead>
                <tbody>
                    {html_filas}
                </tbody>
            </table>
        </div>
    """
    st.markdown(tabla_html, unsafe_allow_html=True)
else:
    st.info("Aún no hay registros en la base de datos.")



# PARTE FINAL: TEXTOS EXPLICATIVOS DEL PROYECTO Y MI LINKEDIN

# Función para convertir imágenes locales a base64 

def img_to_base64(img_path):
    import base64
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except: return ""

# Foto perfil

foto_perfil = img_to_base64("img/yo.jpg")

# Apartados explicativos en formato markdown
st.markdown("---")
st.markdown(f"""
    <div id="modal-proyecto" class="modal-overlay" style="display: block; position: relative; background: none; height: auto;">
            <div class="modal-content-static">
                <div class="modal-header-neo">SOBRE EL PROYECTO</div>
                <div class="modal-body-neo">
             <div style="text-align: center; margin-top: 15px;">
                    <p>Arquitectura <b>CNN</b> en <b>PyTorch</b> para análisis de radiografías pulmonares.
                    El laboratorio asociado al desarrollo y entrenamiento de la red neuronal es el siguiente:</p>
            <a href="https://colab.research.google.com/drive/13KDYPImDFdXwgClRaYG1I1r2xekvKTa5?usp=sharing" target="_blank" class="btn-link-neo">
            LABORATORIO DE CREACIÓN Y ENTRENAMIENTO
            </a>
            </div>
                    <hr class="hr-neo">
                    <p>El objetivo del proyecto es poder aplicar una arquitectura de una red neuronal entrenada a un sitio web. 
                    El proyecto es educativo y funciona correctamente con las imágenes del siguiente enlace:</p>
                    <div style="text-align: center; margin-top: 15px;">
                        <a href="https://drive.google.com/drive/u/0/folders/1NCFivZMemajwFNv-4JI5GJG73ySPN9A6" target="_blank" class="btn-link-neo">
                            VER IMÁGENES EN GOOGLE DRIVE
                        </a>
                    </div>
                </div>
            </div>
        </div>

<div id="modal-perfil" class="modal-content-static" style="margin-top: 20px;">
        <div class="modal-header-neo">SOBRE EL DESARROLLADOR</div>
        <div class="modal-body-neo">
            <div class="perfil-container">
                <div class="perfil-foto">
                    <img src="data:image/jpeg;base64,{foto_perfil}" alt="Martín Zwarycz">
                </div>
                <div class="perfil-texto">
                    <p>Me llamo <b>Martín Zwarycz</b>, estudiante a 5 materias de ser analista en informática y 25 de ser licenciado en informática.</p>
                    <p>Soy un apasionado por el aprendizaje en general, con mucho interés en el desarrollo de aplicaciones IA.</p>
                    <div style="text-align: center; margin-top: 15px;">
                        <a href="https://www.linkedin.com/in/martín-zwarycz-95aab9213/" target="_blank" class="btn-link-neo">LINKEDIN</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)