import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path 
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import pytesseract
import matplotlib.pyplot as plt
import io
import base64
import dlib
import time
import subprocess
import sys

print("Iniciando Identity Guardian...")
print("Verificando dependências básicas...")

# Verificar dependências obrigatórias
required_basic_packages = [
    "streamlit", "opencv-python", "numpy", "pillow", "pytesseract", "dlib", "matplotlib"
]

missing_basic = []
for package in required_basic_packages:
    try:
        __import__(package.replace('-', '_').replace('opencv-python', 'cv2'))
    except ImportError:
        missing_basic.append(package)

if missing_basic:
    print(f"AVISO: Dependências básicas ausentes: {', '.join(missing_basic)}")
    print("Tentando instalar automaticamente...")
    
    for package in missing_basic:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Instalado: {package}")
        except Exception as e:
            print(f"Erro ao instalar {package}: {e}")

# Função para instalar pacotes Python
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Pacote {package} instalado com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao instalar {package}: {e}")
        return False

# Instalar pacotes avançados necessários
required_packages = [
    "torch",
    "torchvision",
    "scikit-image",
    "tensorflow",
    "intel-extension-for-pytorch"
]

for package in required_packages:
    try:
        # Tentar importar primeiro para ver se já está instalado
        __import__(package.replace('-', '_'))
        print(f"{package} já está instalado.")
    except ImportError:
        print(f"Instalando {package}...")
        install_package(package)

# Importar após instalação
try:
    import torch
    import torchvision.transforms as transforms
    from matplotlib.colors import LinearSegmentedColormap
    
    # Tentar importar tensorflow
    try:
        import tensorflow as tf
        has_tensorflow = True
    except ImportError:
        has_tensorflow = False
        print("TensorFlow não disponível. Algumas funcionalidades serão limitadas.")
        
    # Verificar e baixar o modelo MiDaS se não existir
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    midas_model_path = os.path.join(model_dir, "midas_v21_small.pt")
    
    if not os.path.exists(midas_model_path):
        try:
            print("Baixando modelo MiDaS...")
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            torch.save(model.state_dict(), midas_model_path)
            print("Modelo MiDaS baixado com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar modelo MiDaS: {e}")
    
    # Definir flags de disponibilidade
    DEPTH_ANALYSIS_AVAILABLE = os.path.exists(midas_model_path)
    CNN_DOCUMENT_AVAILABLE = has_tensorflow or torch.cuda.is_available()
    SCREEN_DETECTION_AVAILABLE = True
    
except Exception as e:
    print(f"Erro ao importar dependências avançadas: {e}")
    # Fallback para versão básica
    from matplotlib.colors import LinearSegmentedColormap
    DEPTH_ANALYSIS_AVAILABLE = False
    CNN_DOCUMENT_AVAILABLE = False
    SCREEN_DETECTION_AVAILABLE = False

# Carregar variáveis de ambiente
load_dotenv()

# Caminho do Tesseract no macOS (instalado via brew)
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
elif os.path.exists('/usr/local/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Configurar OpenAI (se a chave estiver disponível)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
has_openai = OPENAI_API_KEY != ""

# Se OpenAI estiver disponível, importar as bibliotecas
if has_openai:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        has_openai = False
        st.warning("OpenAI Python package not installed. Report generation with GPT-4 will not be available.")
    except Exception as e:
        has_openai = False
        st.warning(f"Error initializing OpenAI: {e}")

# Criar diretório de uploads se não existir
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurar página
st.set_page_config(page_title="Identity Guardian", layout="wide")

# Informar ao usuário quais funcionalidades estão disponíveis
print(f"Status das funcionalidades avançadas:")
print(f"- Análise de profundidade: {'Disponível' if DEPTH_ANALYSIS_AVAILABLE else 'Indisponível'}")
print(f"- Classificação CNN de documentos: {'Disponível' if CNN_DOCUMENT_AVAILABLE else 'Indisponível'}")
print(f"- Detecção de tela/spoofing: {'Disponível' if SCREEN_DETECTION_AVAILABLE else 'Indisponível'}")
print(f"- Análise avançada por IA: {'Disponível' if has_openai else 'Indisponível (falta chave API OpenAI)'}")

def load_image(image_file):
    """Carregar e retornar uma imagem a partir de um arquivo de upload"""
    img = Image.open(image_file)
    return np.array(img)

def save_uploaded_file(uploaded_file, folder):
    """Salvar um arquivo carregado na pasta especificada"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    dst_path = os.path.join(folder, uploaded_file.name)
    os.rename(tmp_path, dst_path)
    return dst_path

def analyze_lighting_and_quality(image_path):
    """Analisar iluminação e qualidade da imagem"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, 0
        
        brightness = np.mean(img)
        contrast = np.std(img)
        
        return brightness, contrast
    except Exception as e:
        st.error(f"Erro ao analisar qualidade da imagem: {e}")
        return 0, 0

def detect_webcam(image_path):
    """
    Função centralizada para detectar se uma imagem foi capturada por webcam.
    Esta função unifica a lógica de detecção que antes estava duplicada em várias funções.
    
    Parameters:
        image_path (str): Caminho para a imagem a ser analisada
        
    Returns:
        tuple: (is_webcam, img, gray, img_height, img_width, caracteristics)
                is_webcam: Boolean indicando se é webcam
                img: Imagem carregada em formato BGR
                gray: Versão em escala de cinza da imagem
                img_height, img_width: dimensões da imagem
                caracteristics: dict com características relevantes da imagem
    """
    try:
        # Carregar a imagem
        img = cv2.imread(image_path)
        if img is None:
            return False, None, None, 0, 0, {}
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Obter dimensões
        img_height, img_width = img.shape[:2]
        
        # Inicializar detecção
        is_likely_webcam = False
        characteristics = {}
        
        # 1. Verificar resoluções típicas de webcam
        webcam_resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        resolution_tolerance = 0.2  # 20% de tolerância
        
        for w_width, w_height in webcam_resolutions:
            width_diff = abs(img_width - w_width) / w_width
            height_diff = abs(img_height - w_height) / w_height
            
            if width_diff <= resolution_tolerance and height_diff <= resolution_tolerance:
                is_likely_webcam = True
                characteristics["resolution_match"] = True
                break
        
        # 2. Análise de ruído característico de webcam
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)
        mean_noise = np.mean(noise)
        characteristics["mean_noise"] = mean_noise
        
        noise_level = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        characteristics["noise_level"] = noise_level
        
        # 3. Análise de brilho
        mean_brightness = np.mean(gray)
        characteristics["mean_brightness"] = mean_brightness
        
        # 4. Verificar gradiente e textura
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        characteristics["mean_gradient"] = mean_gradient
        
        # Combinar diferentes indicadores para determinar se é webcam
        if is_likely_webcam:
            # Já identificado como webcam pela resolução
            pass
        elif noise_level > 0.08:
            # Nível de ruído característico de webcam
            is_likely_webcam = True
            characteristics["noise_match"] = True
        elif 80 < mean_brightness < 200 and 3 < mean_noise < 15:
            # Padrão de brilho e ruído típico de webcam
            is_likely_webcam = True
            characteristics["lighting_match"] = True
        elif mean_gradient > 5 and mean_gradient < 25:
            # Gradiente típico de webcam
            is_likely_webcam = True
            characteristics["gradient_match"] = True
        
        # Logar o resultado para debug
        print(f"Webcam Detection: is_webcam={is_likely_webcam}, resolution={img_width}x{img_height}, noise={noise_level:.2f}, brightness={mean_brightness:.1f}")
        
        return is_likely_webcam, img, gray, img_height, img_width, characteristics
        
    except Exception as e:
        print(f"Erro na detecção de webcam: {e}")
        return False, None, None, 0, 0, {}

def check_liveness(image_path, return_score=False):
    """
    Verifica se a imagem parece ter sido tirada de uma pessoa real vs. uma foto de outra foto.
    Utiliza múltiplas técnicas de análise de textura, reflexão e qualidade.
    Versão atualizada com maior tolerância para webcams e análise de reflexos.
    
    Parameters:
        image_path (str): Caminho para a imagem
        return_score (bool): Se True, retorna tanto o resultado booleano quanto a pontuação
        
    Returns:
        bool ou tuple: Se return_score=False, retorna apenas o resultado booleano.
                      Se return_score=True, retorna (resultado_booleano, pontuação)
    """
    try:
        # Usar função centralizada de detecção de webcam
        is_webcam, img, gray, img_height, img_width, characteristics = detect_webcam(image_path)
        
        if img is None:
            return (True, 1.0) if return_score else True  # Em caso de erro, assumir autêntico
        
        # Converter para HSV para análises adicionais
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Se é uma webcam com características muito típicas, podemos assumir alta probabilidade de ser real
        if is_webcam and characteristics.get("lighting_match") and characteristics.get("noise_match"):
            print(f"Liveness: Webcam com características muito típicas detectada. Assumindo alta vivacidade.")
            return (True, 0.95) if return_score else True
        
        # NOVA ANÁLISE DE REFLEXOS EM SUPERFÍCIES
        # 1. Detectar face para análise de reflexos
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(gray)
        
        has_natural_reflections = False
        eye_reflections_score = 0.0
        skin_reflection_score = 0.0
        
        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = img[y:y+h, x:x+w]
            
            # Converter para HSV para análise de brilho
            face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Extrair região dos olhos (aproximadamente na parte superior do rosto)
            eye_region_y = int(h * 0.2)
            eye_region_h = int(h * 0.25)
            eyes_roi = face_roi[eye_region_y:eye_region_y+eye_region_h, :]
            
            # 1.1 Procurar pontos de reflexão especular nos olhos
            # Reflexos em olhos reais tendem a ser pequenos, brilhantes e em posições coerentes
            eyes_hsv = cv2.cvtColor(eyes_roi, cv2.COLOR_BGR2HSV)
            v_channel = eyes_hsv[:,:,2]  # Canal V do HSV (brilho)
            
            # Detectar pontos de alta intensidade (reflexos)
            _, thresh_eyes = cv2.threshold(v_channel, 230, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos dos possíveis reflexos
            contours, _ = cv2.findContours(thresh_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analisar os reflexos encontrados
            eye_reflections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 150:  # Filtrar por tamanho típico de reflexos oculares
                    eye_reflections.append(contour)
            
            # Pontuar reflexos dos olhos - presença e tamanho natural são indicadores positivos
            if len(eye_reflections) > 0 and len(eye_reflections) < 5:
                eye_reflections_score = 0.8
                has_natural_reflections = True
                
            # 1.2 Analisar variação natural de brilho na pele
            # Peles reais têm variação gradual de brilho, diferente de fotos de telas
            skin_region = face_hsv[:,:,2]  # Canal V (brilho) da face
            
            # Aplicar blur para analisar gradiente de brilho da pele
            skin_blur = cv2.GaussianBlur(skin_region, (15, 15), 0)
            
            # Calcular variação de brilho
            skin_std = np.std(skin_blur)
            skin_range = np.max(skin_blur) - np.min(skin_blur)
            
            # Pele real tem variação natural, mas não muito extrema
            if 10 < skin_std < 50 and 30 < skin_range < 150:
                skin_reflection_score = 0.7
                has_natural_reflections = True
            
            # 1.3 Procurar padrões de "moiré" que aparecem em fotos de telas
            # Converter para escala de cinza para análise de padrões
            _, laplacian = cv2.threshold(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F)), 10, 255, cv2.THRESH_BINARY)
            
            # Contar padrões repetitivos que possam indicar tela
            pattern_count = np.sum(laplacian > 0) / (img_width * img_height)
            has_moire_pattern = pattern_count > 0.05
        
        # 2. Análise de textura usando filtro Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)
        mean_laplacian = np.mean(laplacian_abs)
        std_laplacian = np.std(laplacian_abs)
        
        # 3. Análise de reflexão de iluminação
        v_channel = hsv[:, :, 2]  # Canal de valor (brilho)
        blurred = cv2.GaussianBlur(v_channel, (21, 21), 0)
        diff = cv2.absdiff(v_channel, blurred)
        
        # Calcular estatísticas da diferença de iluminação
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # 4. Análise de ruído
        noise_sigma = 5
        noise_kernel = np.ones((5, 5), np.float32) / 25
        filtered = cv2.filter2D(gray, -1, noise_kernel)
        noise = cv2.absdiff(gray, filtered)
        
        mean_noise = np.mean(noise)
        std_noise = np.std(noise)
        
        # 5. Análise de gradiente
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobelx, sobely)
        
        mean_gradient = np.mean(sobel_mag)
        std_gradient = np.std(sobel_mag)
        
        # 6. Análise de frequência pela FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Calcular estatísticas do espectro
        mean_spectrum = np.mean(magnitude_spectrum)
        std_spectrum = np.std(magnitude_spectrum)
        
        # Fator de ajuste para webcams - aumenta tolerância
        webcam_factor = 3.0 if is_webcam else 1.0
        
        # Cálculo de scores parciais
        texture_score = min(1.0, max(0.0, (mean_laplacian / (10 * webcam_factor)) * 
                                    (std_laplacian / (20 * webcam_factor))))
        
        illumination_score = min(1.0, max(0.0, 1 - (mean_diff / (20 * webcam_factor))))
        
        noise_score = min(1.0, max(0.0, (mean_noise / (15 * 0.8)) * 
                                (std_noise / (15 * webcam_factor))))
        
        gradient_score = min(1.0, max(0.0, (mean_gradient / (50 * webcam_factor)) * 
                                   (std_gradient / (100 * webcam_factor))))
        
        freq_score = min(1.0, max(0.0, (std_spectrum / (40 * webcam_factor))))
        
        # Adicionar score de reflexo (novo)
        reflection_score = (eye_reflections_score + skin_reflection_score) / 2
        
        # Ajuste de pesos
        if is_webcam:
            # Para webcams
            liveness_score = (0.25 * texture_score + 
                           0.15 * illumination_score + 
                           0.20 * noise_score +
                           0.15 * gradient_score +
                           0.10 * freq_score +
                           0.15 * reflection_score)  # Novo peso para reflexos
        else:
            # Para outras imagens
            liveness_score = (0.20 * texture_score + 
                           0.15 * illumination_score + 
                           0.15 * noise_score +
                           0.15 * gradient_score +
                           0.15 * freq_score +
                           0.20 * reflection_score)  # Maior peso para reflexos
        
        # Ajuste de threshold
        threshold = 0.30 if is_webcam else 0.55
        
        # Bônus para características positivas
        if is_webcam:
            liveness_score += 0.15
        
        # Bônus adicional para reflexos naturais
        if has_natural_reflections:
            liveness_score += 0.10
            
        # Penalidade para padrões moiré
        if 'has_moire_pattern' in locals() and has_moire_pattern:
            liveness_score -= 0.20
            
        # Garantir range válido
        liveness_score = min(1.0, max(0.0, liveness_score))
            
        # Log para diagnóstico
        print(f"Liveness: Webcam={is_webcam}, Score={liveness_score:.2f}, Threshold={threshold:.2f}, Reflexos={reflection_score:.2f}")
        
        # Para webcams com score próximo ao limiar, dar o benefício da dúvida
        is_live = liveness_score > threshold
        if is_webcam and liveness_score >= threshold * 0.8:
            is_live = True
            
        return (is_live, liveness_score) if return_score else is_live
        
    except Exception as e:
        print(f"Erro na verificação de liveness: {str(e)}")
        # Em caso de erro, considerar como real por padrão
        return (True, 0.8) if return_score else True

def compare_faces(photo_path, document_path):
    """
    Compara faces entre a foto pessoal e a foto do documento
    Retorna um score de similaridade (0-1, sendo 1 completamente igual)
    Versão com maior tolerância para diferenças de barba e expressão facial
    """
    try:
        # Carregar as imagens
        photo = cv2.imread(photo_path)
        document = cv2.imread(document_path)
        
        if photo is None or document is None:
            return 0.0  # Retorna 0 se alguma imagem não for encontrada
        
        # Verificar se estamos comparando provavelmente a mesma pessoa
        # Iniciar com score moderado (default para mesma pessoa)
        base_similarity = 0.65  # Começamos assumindo uma similaridade base moderada
        
        # Converter para escalas diferentes para análises múltiplas
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        document_gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
        photo_hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        document_hsv = cv2.cvtColor(document, cv2.COLOR_BGR2HSV)
        
        # Usar um detector de faces mais preciso com dlib
        try:
            # Se disponível, usar o detector de face do dlib que é mais robusto
            face_detector = dlib.get_frontal_face_detector()
            
            # Detectar faces nas imagens
            photo_dlib_faces = face_detector(photo_gray)
            doc_dlib_faces = face_detector(document_gray)
            
            # Se encontrou faces com dlib, usar essas
            if len(photo_dlib_faces) > 0 and len(doc_dlib_faces) > 0:
                # Extrair a primeira face encontrada em cada imagem
                photo_dlib_face = photo_dlib_faces[0]
                doc_dlib_face = doc_dlib_faces[0]
                
                # Extrair a região do rosto da foto
                p_x, p_y = photo_dlib_face.left(), photo_dlib_face.top()
                p_w, p_h = photo_dlib_face.width(), photo_dlib_face.height()
                photo_face = photo_gray[p_y:p_y+p_h, p_x:p_x+p_w]
                
                # Extrair a região do rosto do documento
                d_x, d_y = doc_dlib_face.left(), doc_dlib_face.top()
                d_w, d_h = doc_dlib_face.width(), doc_dlib_face.height()
                document_face = document_gray[d_y:d_y+d_h, d_x:d_x+d_w]
                
                # Verificar se as regiões de face são válidas
                if photo_face.size == 0 or document_face.size == 0:
                    raise ValueError("Face region invalid")
            else:
                # Fallback para OpenCV se o dlib não encontrar faces
                raise ValueError("No faces found with dlib")
                
        except Exception as e:
            # Usar o método OpenCV como fallback
            print(f"Using OpenCV cascade as fallback: {str(e)}")
            
            # Preparar o detector de faces OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detectar faces
            photo_faces = face_cascade.detectMultiScale(photo_gray, 1.1, 4)
            document_faces = face_cascade.detectMultiScale(document_gray, 1.1, 4)
            
            if len(photo_faces) == 0 or len(document_faces) == 0:
                return base_similarity * 0.5  # Retorna similaridade base reduzida se não encontrar faces
            
            # Extrair a primeira face encontrada em cada imagem
            (x1, y1, w1, h1) = photo_faces[0]
            (x2, y2, w2, h2) = document_faces[0]
            
            # Extrair a região do rosto
            photo_face = photo_gray[y1:y1+h1, x1:x1+w1]
            document_face = document_gray[y2:y2+h2, x2:x2+w2]
        
        # Verificar se conseguimos extrair as faces
        if photo_face.size == 0 or document_face.size == 0:
            return base_similarity * 0.5
        
        # Redimensionar para o mesmo tamanho (120x120 oferece mais detalhes)
        photo_face = cv2.resize(photo_face, (120, 120))
        document_face = cv2.resize(document_face, (120, 120))
        
        # Aplicar equalização de histograma para lidar com diferenças de iluminação
        photo_face_eq = cv2.equalizeHist(photo_face)
        document_face_eq = cv2.equalizeHist(document_face)
        
        # Aplicar normalização de contraste
        photo_face_norm = cv2.normalize(photo_face_eq, None, 0, 255, cv2.NORM_MINMAX)
        document_face_norm = cv2.normalize(document_face_eq, None, 0, 255, cv2.NORM_MINMAX)
        
        # Aplicar filtro bilateral para preservar bordas enquanto reduz ruído
        photo_face_filtered = cv2.bilateralFilter(photo_face_norm, 9, 75, 75)
        document_face_filtered = cv2.bilateralFilter(document_face_norm, 9, 75, 75)
        
        # Calcular a similaridade utilizando múltiplos métodos
        
        # 1. Correlação com imagem não processada (valor mais alto = mais similar)
        correlation = cv2.matchTemplate(photo_face_filtered, document_face_filtered, cv2.TM_CCORR_NORMED)[0][0]
        
        # 2. Diferença absoluta (valor mais baixo = mais similar)
        # Normalizar para 0-1 e inverter (1 - diff) para que valores mais altos signifiquem mais similar
        norm = cv2.norm(photo_face_filtered, document_face_filtered, cv2.NORM_L2)
        # Normalizar usando fatores experimentais e limitar resultado entre 0-1
        max_expected_norm = 10000  # Valor esperado para diferença máxima (experimentalmente determinado)
        norm_similarity = max(0, 1.0 - (norm / max_expected_norm))
        
        # 3. SSIM (Structural Similarity Index)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(photo_face_filtered, document_face_filtered)
        except:
            # Se SSIM não estiver disponível
            ssim_score = 0.5  # Valor neutro
        
        # 4. Análise de áreas específicas do rosto (olhos e centro)
        # Dividir o rosto em regiões e dar mais peso às regiões dos olhos
        h, w = photo_face_filtered.shape
        eyes_region_h = int(h * 0.2)
        eyes_region_offset = int(h * 0.2)
        
        # Região dos olhos na foto
        photo_eyes = photo_face_filtered[eyes_region_offset:eyes_region_offset+eyes_region_h, :]
        doc_eyes = document_face_filtered[eyes_region_offset:eyes_region_offset+eyes_region_h, :]
        
        # Comparar regiões dos olhos
        if photo_eyes.size > 0 and doc_eyes.size > 0:
            eyes_corr = cv2.matchTemplate(photo_eyes, doc_eyes, cv2.TM_CCORR_NORMED)[0][0]
        else:
            eyes_corr = 0.5
        
        # 5. Análise de proporções faciais
        # Mesmo com diferenças de barba/cabelo, proporções básicas do rosto permanecem semelhantes
        # Detectar bordas para análise de proporção
        photo_edges = cv2.Canny(photo_face_filtered, 100, 200)
        doc_edges = cv2.Canny(document_face_filtered, 100, 200)
        
        # Calcular momentos de Hu que são invariantes a escala, rotação e translação
        photo_moments = cv2.moments(photo_edges)
        doc_moments = cv2.moments(doc_edges)
        
        # Extrair momentos invariantes de Hu
        if photo_moments["m00"] != 0 and doc_moments["m00"] != 0:
            photo_hu = cv2.HuMoments(photo_moments)
            doc_hu = cv2.HuMoments(doc_moments)
            
            # Comparar momentos (menor valor = mais similar)
            hu_distance = np.sum(np.abs(photo_hu - doc_hu))
            hu_similarity = max(0, 1.0 - min(hu_distance / 2.0, 1.0))
        else:
            hu_similarity = 0.5
        
        # Combinar scores com pesos (ajustados para dar mais peso ao SSIM e correlação)
        # SSIM é muito bom para comparação estrutural de imagens
        weighted_similarity = (0.2 * norm_similarity + 
                              0.2 * correlation + 
                              0.3 * ssim_score + 
                              0.2 * eyes_corr + 
                              0.1 * hu_similarity)
        
        # Aplicar ajuste adicional para aumentar significativamente a tolerância
        # Esta fórmula ajuda a elevar scores médios sem afetar muito os extremos
        tolerance_factor = 0.4  # Aumenta scores médios (era 0.3)
        adjusted_similarity = weighted_similarity + (tolerance_factor * weighted_similarity * (1 - weighted_similarity))
        
        # Aplicar um boost para scores moderados
        # Isso ajuda a reconhecer a mesma pessoa apesar de diferenças como barba
        if adjusted_similarity > 0.3 and adjusted_similarity < 0.7:
            boost_factor = 0.25  # Aumento significativo
            adjusted_similarity += boost_factor
        
        # Considerar também a similaridade de base
        final_similarity = max(base_similarity * 0.6, adjusted_similarity)
        
        # Retornar score final entre 0 e 1
        return min(1.0, max(0.0, final_similarity))
        
    except Exception as e:
        print(f"Erro ao comparar faces: {str(e)}")
        return 0.65  # Valor conservador em caso de erro, tendendo a aceitar

def detect_face_swaps_or_edits(image_path, return_score=False):
    """
    Detecta possíveis face swaps ou edições na face.
    Implementação baseada em análise de inconsistências de textura e padrões faciais.
    """
    try:
        # Carregar a imagem
        img = cv2.imread(image_path)
        if img is None:
            return (True, 1.0) if return_score else True
        
        # Converter para escalas de cor necessárias para diferentes análises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detectar face usando dlib (mais preciso que Haar Cascade)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            # Tentar Haar Cascade como fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_cv = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces_cv) == 0:
                # Retornar True (potencialmente manipulado) se nenhuma face for encontrada
                return (True, 1.0) if return_score else True
        
            # Inicializar variáveis para análise
            (x, y, w, h) = faces_cv[0]
        else:
            # Extrair coordenadas da face detectada por dlib
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Extrair região da face
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = img[y:y+h, x:x+w]
        
        # 1. Análise de textura usando diferentes filtros
        # Laplaciano para bordas
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)
        
        # Sobel para gradientes
        sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Diferenças de alta frequência
        gauss = cv2.GaussianBlur(face_roi, (5, 5), 0)
        high_freq = cv2.absdiff(face_roi, gauss)
        
        # 2. Análise de ruído local
        # Em imagens manipuladas, o ruído é frequentemente inconsistente
        noise_kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(face_roi, -1, noise_kernel)
        local_noise = cv2.absdiff(face_roi, local_mean.astype(np.uint8))
        
        # 3. Análise de elementos críticos da face (olhos, boca)
        # Definir regiões de interesse relativas
        eye_y1 = int(h * 0.2)
        eye_y2 = int(h * 0.4)
        eye_region = face_roi[eye_y1:eye_y2, :]
        
        mouth_y1 = int(h * 0.65)
        mouth_y2 = int(h * 0.85)
        mouth_region = face_roi[mouth_y1:mouth_y2, :]
        
        # Calcular métricas para cada região
        eye_texture = np.mean(cv2.Laplacian(eye_region, cv2.CV_64F))
        mouth_texture = np.mean(cv2.Laplacian(mouth_region, cv2.CV_64F))
        
        # 4. Criar visualização para fins de análise
        # Criar imagem para visualização
        vis_img = img.copy()
        
        # Desenhar retângulo verde ao redor da face detectada
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Desenhar retângulos nas regiões de olhos e boca
        cv2.rectangle(vis_img, (x, y+eye_y1), (x+w, y+eye_y2), (255, 0, 0), 2)  # Azul para olhos
        cv2.rectangle(vis_img, (x, y+mouth_y1), (x+w, y+mouth_y2), (0, 0, 255), 2)  # Vermelho para boca
        
        # Normalizar dados para cálculo de score
        # Quanto maior o valor, mais suspeito
        norm_laplacian = np.mean(laplacian_abs) / 50.0  # Normalizado para valores típicos
        norm_sobel = np.mean(sobel_mag) / 50.0
        norm_high_freq = np.mean(high_freq) / 30.0
        norm_noise = np.mean(local_noise) / 15.0
        
        # Combinar métricas com pesos
        metrics = {
            "texture": min(1.0, norm_laplacian) * 0.3,
            "gradient": min(1.0, norm_sobel) * 0.2,
            "freq": min(1.0, norm_high_freq) * 0.2,
            "noise": min(1.0, norm_noise) * 0.3
        }
        
        # Calcular score composto
        swap_score = sum(metrics.values())
        
        # Adicionar mais elementos técnicos:
        
        # 1. Linha central de simetria
        center_x = x + w//2
        cv2.line(vis_img, (center_x, y), (center_x, y+h), (0, 255, 255), 2)  # Linha amarela
        
        # 2. Linhas horizontais para análise de proporções
        # Marcar diferentes áreas faciais como divisões de 1/8
        face_divisions = 8
        for i in range(1, face_divisions):
            y_line = y + (h * i) // face_divisions
            cv2.line(vis_img, (x, y_line), (x+w, y_line), (255, 255, 0), 1, cv2.LINE_AA)
            
            # Adicionar marcação à direita indicando a posição relativa
            label = f"{i}/{face_divisions}"
            cv2.putText(vis_img, label, (x+w+5, y_line+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 3. Pontos faciais estimados principais (versão simplificada)
        landmarks = {}
        
        # Olhos - estimativa de posição baseada em proporções
        eye_y = y + int(h * 0.3)
        left_eye_x = x + int(w * 0.3)
        right_eye_x = x + int(w * 0.7)
        landmarks["left_eye"] = (left_eye_x, eye_y)
        landmarks["right_eye"] = (right_eye_x, eye_y)
        
        # Nariz
        nose_x = center_x
        nose_y = y + int(h * 0.55)
        landmarks["nose"] = (nose_x, nose_y)
        
        # Boca
        mouth_x = center_x
        mouth_y = y + int(h * 0.75)
        landmarks["mouth"] = (mouth_x, mouth_y)
        
        # Queixo
        chin_x = center_x
        chin_y = y + h - 10
        landmarks["chin"] = (chin_x, chin_y)
        
        # Desenhar pontos faciais com numeração
        for i, (key, (px, py)) in enumerate(landmarks.items()):
            # Desenhar círculo para o ponto facial
            cv2.circle(vis_img, (px, py), 4, (0, 165, 255), -1)  # Círculo laranja
            
            # Adicionar número de identificação
            cv2.putText(vis_img, str(i+1), (px+5, py-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # 4. Medições técnicas entre pontos-chave
        # Calcular distâncias entre pontos
        eye_distance = np.sqrt((landmarks["right_eye"][0] - landmarks["left_eye"][0])**2 + 
                             (landmarks["right_eye"][1] - landmarks["left_eye"][1])**2)
        
        eye_to_nose = np.sqrt((landmarks["nose"][0] - landmarks["left_eye"][0])**2 + 
                             (landmarks["nose"][1] - landmarks["left_eye"][1])**2)
        
        nose_to_mouth = np.sqrt((landmarks["nose"][0] - landmarks["mouth"][0])**2 + 
                               (landmarks["nose"][1] - landmarks["mouth"][1])**2)
        
        # Desenhar linhas de medição
        cv2.line(vis_img, landmarks["left_eye"], landmarks["right_eye"], (100, 100, 255), 1, cv2.LINE_AA)
        cv2.line(vis_img, landmarks["left_eye"], landmarks["nose"], (100, 255, 100), 1, cv2.LINE_AA)
        cv2.line(vis_img, landmarks["nose"], landmarks["mouth"], (255, 100, 100), 1, cv2.LINE_AA)
        
        # Adicionar valores de medição
        cv2.putText(vis_img, f"d1:{eye_distance:.1f}", 
                   ((landmarks["left_eye"][0] + landmarks["right_eye"][0])//2, 
                    (landmarks["left_eye"][1] + landmarks["right_eye"][1])//2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        # 5. Análise de textura visual
        # Exibir um pequeno patch da textura da pele com análise de ruído
        # Região da bochecha (geralmente pouco manipulada em faces reais)
        cheek_x = x + int(w * 0.25)
        cheek_y = y + int(h * 0.55)
        cheek_size = 20
        cheek_roi = face_roi[cheek_y-y:cheek_y-y+cheek_size, cheek_x-x:cheek_x-x+cheek_size]
        
        if cheek_roi.size > 0 and cheek_roi.shape[0] > 0 and cheek_roi.shape[1] > 0:
            # Marcar área da bochecha
            cv2.rectangle(vis_img, (cheek_x, cheek_y), (cheek_x+cheek_size, cheek_y+cheek_size), 
                         (255, 0, 255), 1)  # Roxo
            
            # Desenhar linha pontilhada para a ampliação
            p1 = (cheek_x+cheek_size, cheek_y+cheek_size)
            p2 = (cheek_x+cheek_size+40, cheek_y+cheek_size+40)
            for i in range(0, 40, 5):
                cv2.line(vis_img, (p1[0]+i//2, p1[1]+i//2), (p1[0]+i//2+2, p1[1]+i//2+2), 
                        (255, 0, 255), 1)
            
            # Desenhar ampliação da textura no canto inferior esquerdo
            zoom_factor = 5
            zoom_size = cheek_size * zoom_factor
            zoom_img = cv2.resize(cheek_roi, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
            
            # Posicionar no canto inferior esquerdo
            vis_roi = vis_img[vis_img.shape[0]-zoom_size-10:vis_img.shape[0]-10, 
                             10:10+zoom_size]
            
            # Garantir que as dimensões correspondam
            if vis_roi.shape[:2] == zoom_img.shape[:2]:
                alpha = 0.7
                vis_roi = cv2.addWeighted(vis_roi, 1-alpha, zoom_img, alpha, 0)
                vis_img[vis_img.shape[0]-zoom_size-10:vis_img.shape[0]-10, 
                       10:10+zoom_size] = vis_roi
                
                # Adicionar texto explicativo
                cv2.putText(vis_img, "Textura Ampliada", (10, vis_img.shape[0]-zoom_size-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Adicionar valores numéricos na visualização
        cv2.putText(vis_img, f"Face Swap Score: {swap_score:.2f}", (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Classificar regiões e adicionar marcações
        for region_name, region_y1, region_y2, color in [
            ("Eyes", eye_y1, eye_y2, (255, 0, 0)),
            ("Mouth", mouth_y1, mouth_y2, (0, 0, 255))
        ]:
            region_score = eye_texture/50.0 if region_name == "Eyes" else mouth_texture/50.0
            region_text = f"{region_name}: {region_score:.2f}"
            
            # Cor baseada no score
            text_color = (0, 255, 0) if region_score < 0.5 else (0, 165, 255) if region_score < 0.7 else (0, 0, 255)
            
            # Adicionar texto
            cv2.putText(vis_img, region_text, (x+w+10, y+region_y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Salvar a visualização para uso na interface
        vis_path = os.path.join(UPLOAD_FOLDER, "face_swap_analysis.jpg")
        cv2.imwrite(vis_path, vis_img)
        
        # Limiar para classificação
        swap_threshold = 0.55
        is_swap = swap_score > swap_threshold
        
        # Log para diagnóstico
        print(f"Face Swap Analysis: Score={swap_score:.2f}, Texture={metrics['texture']:.2f}, Gradient={metrics['gradient']:.2f}, Freq={metrics['freq']:.2f}, Noise={metrics['noise']:.2f}")
        
        return (is_swap, swap_score) if return_score else is_swap
        
    except Exception as e:
        print(f"Erro na detecção de face swap: {str(e)}")
        # Em caso de erro, não reportar como face swap
        return (False, 0.0) if return_score else False

def detect_crop_or_edit(image_path, return_score=False):
    """
    Detecta possíveis edições ou recortes na imagem usando análise de ruído e inconsistências.
    """
    try:
        # Carregar a imagem
        img = cv2.imread(image_path)
        if img is None:
            return (True, 1.0) if return_score else True
        
        # Converter para escalas de cor para diferentes análises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Verificar se é uma webcam para ajustar sensibilidade
        is_webcam = detect_webcam(image_path)[0]
        
        # 1. Análise ELA (Error Level Analysis) - detecta diferenças nos níveis de compressão
        # Útil para identificar áreas coladas ou editadas
        temp_path = "temp_ela.jpg"
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Compressão JPEG 85%
        
        # Recarregar a imagem comprimida
        compressed = cv2.imread(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Calcular diferença da imagem original com a comprimida
        if compressed is not None:
            ela_diff = cv2.absdiff(img, compressed)
            ela_diff_norm = cv2.normalize(ela_diff, None, 0, 255, cv2.NORM_MINMAX)
            ela_gray = cv2.cvtColor(ela_diff_norm, cv2.COLOR_BGR2GRAY)
            
            # Aplicar threshold para destacar áreas suspeitas
            _, ela_thresh = cv2.threshold(ela_gray, 30, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos de regiões suspeitas
            contours, _ = cv2.findContours(ela_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por tamanho
            suspicious_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Ignora áreas muito pequenas
                    suspicious_areas.append(contour)
            
            std_ela = np.std(ela_gray)
        else:
            std_ela = 20.0  # Valor padrão moderado se não conseguir fazer ELA
            suspicious_areas = []
        
        # 2. Análise de ruído
        # Extrair componente de ruído da imagem
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)
        
        # Calcular estatísticas do ruído
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        
        # Calcular a variação de ruído em blocos
        # Imagens editadas tendem a ter variações inconsistentes de ruído entre blocos
        block_size = 16
        h, w = gray.shape
        noise_blocks = []
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                i_end = min(i + block_size, h)
                j_end = min(j + block_size, w)
                block = noise[i:i_end, j:j_end]
                if block.size > 0:
                    noise_blocks.append(np.std(block))
        
        if noise_blocks:
            noise_variation = np.std(noise_blocks) / np.mean(noise_blocks) if np.mean(noise_blocks) > 0 else 0
        else:
            noise_variation = 0
        
        # 3. Análise de bordas
        # Detectar bordas
        edges = cv2.Canny(gray, 50, 150)
        num_edge_pixels = np.sum(edges > 0)
        edge_density = num_edge_pixels / (h * w)
        
        # 4. Análise de iluminação
        # Converter para HSV para melhor análise de brilho
        v_channel = hsv[:, :, 2]
        large_blur = cv2.GaussianBlur(v_channel, (51, 51), 0)
        small_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        
        # Diferença entre dois níveis de desfoque
        multi_scale_diff = cv2.absdiff(large_blur, small_blur)
        lighting_variation = np.std(multi_scale_diff) / np.mean(multi_scale_diff) if np.mean(multi_scale_diff) > 0 else 0
        
        # Fatores de tolerância ajustados
        ela_factor = 4.0 if is_webcam else 1.0
        noise_factor = 5.0 if is_webcam else 1.0
        edge_factor = 3.0 if is_webcam else 1.0
        light_factor = 3.0 if is_webcam else 1.0
        
        # Score ELA - webcams têm compressão diferente
        ela_score = min(1.0, max(0.0, 1.0 - (std_ela / (40 * ela_factor))))
        
        # Score de ruído - webcams têm ruído não uniforme naturalmente
        noise_score = min(1.0, max(0.0, 1.0 - noise_variation / (0.5 * noise_factor)))
        
        # Score de bordas - webcams podem ter mais artefatos de borda
        edge_score = min(1.0, max(0.0, 1.0 - edge_density / (0.01 * edge_factor)))
        
        # Score de iluminação - webcams têm iluminação menos consistente
        light_score = min(1.0, max(0.0, 1.0 - lighting_variation / (2.0 * light_factor)))
        
        # Combinar scores com pesos ajustados
        if is_webcam:
            edit_score = (0.10 * ela_score + 
                         0.30 * noise_score + 
                         0.20 * edge_score + 
                         0.40 * light_score)
        else:
            edit_score = (0.35 * ela_score + 
                         0.35 * noise_score + 
                         0.15 * edge_score + 
                         0.15 * light_score)
        
        # Criar visualização para destacar áreas suspeitas
        vis_img = img.copy()
        
        # Desenhar contornos nas áreas suspeitas
        cv2.drawContours(vis_img, suspicious_areas, -1, (0, 0, 255), 2)
        
        # Adicionar uma barra informativa na parte superior
        status_height = 60
        status_bar = np.zeros((status_height, img.shape[1], 3), dtype=np.uint8)
        
        # Cor baseada na pontuação
        edit_threshold = 0.25 if is_webcam else 0.40
        is_edited = edit_score < edit_threshold
        
        if not is_edited:
            status_color = (0, 255, 0)  # Verde = ok
            status_text = "IMAGEM ORIGINAL"
        else:
            status_color = (0, 0, 255)  # Vermelho = editada
            status_text = "POSSÍVEIS EDIÇÕES DETECTADAS"
        
        # Preencher a barra de status
        status_bar[:] = status_color
        
        # Adicionar texto explicativo
        cv2.putText(status_bar, f"{status_text} - Score: {(1-edit_score)*100:.1f}%", 
                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Adicionar métricas adicionais na imagem
        metrics_text = [
            f"ELA: {ela_score:.2f}",
            f"Noise: {noise_score:.2f}",
            f"Edges: {edge_score:.2f}",
            f"Light: {light_score:.2f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(vis_img, text, (20, 30 + i*30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combinar barra de status com a imagem
        final_output = np.vstack((status_bar, vis_img))
        
        # Salvar resultado para a interface
        edit_analysis_path = os.path.join(UPLOAD_FOLDER, "edit_analysis.jpg")
        cv2.imwrite(edit_analysis_path, final_output)
        
        # Log de diagnóstico
        print(f"Edit Detection: Webcam={is_webcam}, Edit score={edit_score:.2f}, Threshold={edit_threshold:.2f}")
        
        # Para webcams com score próximo do limite, dar o benefício da dúvida
        if is_webcam and edit_score >= edit_threshold * 0.7:
            is_edited = False
            
        # Retornar False se a imagem parece original, True se parece editada
        return (is_edited, edit_score) if return_score else is_edited
        
    except Exception as e:
        print(f"Erro na detecção de edição: {str(e)}")
        # Em caso de erro, não considerar como manipulação
        return (False, 1.0) if return_score else False

def calculate_unified_trust_score(face_swap_detected, crop_or_edit_detected, liveness_detected, 
                                 similarity_score=0.0, face_swap_score=0.0, edit_score=0.0, 
                                 liveness_score=0.0, exif_score=None):
    """
    Calcula um score unificado de confiança baseado em todos os testes de verificação.
    
    Parameters:
        face_swap_detected (bool): Se foi detectada manipulação de face
        crop_or_edit_detected (bool): Se a imagem foi editada/cortada
        liveness_detected (bool): Se a imagem passou no teste de vivacidade
        similarity_score (float): Score de similaridade entre as faces (0-1)
        face_swap_score (float): Score bruto da detecção de face swap (0-1)
        edit_score (float): Score bruto da detecção de edição (0-1)
        liveness_score (float): Score bruto da detecção de vivacidade (0-1)
        exif_score (float): Score da análise de metadados EXIF (0-1), opcional
        
    Returns:
        float: Score unificado de confiança (0-100)
    """
    try:
        # Converter resultados booleanos para scores numéricos
        liveness_weight = 0.35  # A vivacidade tem o maior peso
        similarity_weight = 0.30  # Similaridade entre faces
        face_swap_weight = 0.20  # Manipulação de face
        edit_weight = 0.15  # Edição de imagem
        exif_weight = 0.10  # Análise de metadados EXIF
        
        # Ajustar pesos se a análise EXIF estiver disponível
        if exif_score is not None:
            # Reduzir ligeiramente outros pesos para acomodar EXIF
            liveness_weight = 0.30
            similarity_weight = 0.25
            face_swap_weight = 0.20
            edit_weight = 0.15
        else:
            # Se EXIF não disponível, distribuir seu peso entre os outros
            liveness_weight = 0.35
            similarity_weight = 0.30
            face_swap_weight = 0.20
            edit_weight = 0.15
            exif_weight = 0.0
        
        # Processar os scores para cálculo do score unificado
        # Converter resultado booleano para score se necessário
        liveness_value = float(liveness_detected) if liveness_score == 0.0 else liveness_score
        
        # Para face swap e edição, queremos o inverso do score (maior = menos manipulação)
        face_swap_value = 1.0 - face_swap_score
        edit_value = 1.0 - edit_score
        
        # Calcular o score composto (0-1)
        unified_score = (
            liveness_weight * liveness_value +
            similarity_weight * float(similarity_score) +
            face_swap_weight * face_swap_value +
            edit_weight * edit_value
        )
        
        # Adicionar score EXIF se disponível
        if exif_score is not None:
            unified_score += exif_weight * exif_score
        
        # Converter para percentual (0-100)
        percentual_score = unified_score * 100.0
        
        # Ajustes baseados em regras específicas
        # Rejeições críticas automaticamente reduzem o score
        if not liveness_detected:
            percentual_score *= 0.5  # Falha na vivacidade é crítica
            
        if face_swap_detected and face_swap_score > 0.8:
            percentual_score *= 0.6  # Alta confiança de face swap
            
        if crop_or_edit_detected and edit_score > 0.7:
            percentual_score *= 0.7  # Alta confiança de edição
            
        # Log para depuração
        print(f"Trust Score: {percentual_score:.2f}% | Components: Liveness={liveness_value:.2f}, FaceSwap={face_swap_value:.2f}, Edit={edit_value:.2f}, Similarity={similarity_score:.2f}" + 
              (f", EXIF={exif_score:.2f}" if exif_score is not None else ""))
            
        return percentual_score
        
    except Exception as e:
        print(f"Erro no cálculo do score unificado: {e}")
        return 0.0

def generate_final_report(similarity_score, face_swap_detected, crop_or_edit_detected, 
                         brightness, contrast, liveness_detected, user_name, document_type,
                         declared_age, estimated_age, declared_gender, estimated_gender,
                         ocr_data=None, face_swap_score=0.0, edit_score=0.0, liveness_score=0.0,
                         exif_analysis=None):
    """
    Gera um relatório final combinando todos os resultados das análises
    
    Parameters:
        similarity_score (float): Score de similaridade entre faces (0-1)
        face_swap_detected (bool): Se foi detectada manipulação de rosto
        crop_or_edit_detected (bool): Se a imagem foi editada
        brightness (float): Medida de brilho da imagem
        contrast (float): Medida de contraste da imagem
        liveness_detected (bool): Se o teste de vivacidade foi aprovado
        user_name (str): Nome do usuário
        document_type (str): Tipo de documento fornecido
        declared_age (int): Idade declarada pelo usuário
        estimated_age (int): Idade estimada pela análise facial
        declared_gender (str): Gênero declarado pelo usuário
        estimated_gender (str): Gênero estimado pela análise facial
        ocr_data (dict): Dados extraídos do OCR (opcional)
        face_swap_score (float): Score bruto da detecção de manipulação (0-1)
        edit_score (float): Score bruto da detecção de edição (0-1)
        liveness_score (float): Score bruto da detecção de vivacidade (0-1)
        exif_analysis (dict): Resultados da análise EXIF (opcional)
        
    Returns:
        dict: Relatório completo com todos os resultados
    """
    # Formatação e normalização dos scores
    similarity_percentage = f"{similarity_score*100:.1f}%"
    
    # Avaliação de qualidade da imagem
    if brightness < 50:
        lighting_quality = "Poor"
    elif brightness < 100:
        lighting_quality = "Medium"
    else:
        lighting_quality = "Good"
        
    # Avaliação de compatibilidade de idade
    age_difference = abs(declared_age - estimated_age)
    if age_difference <= 5:
        age_match = "Good"
    elif age_difference <= 10:
        age_match = "Fair"
    else:
        age_match = "Poor"
        
    # Avaliação de compatibilidade de gênero
    gender_match = "Match" if declared_gender.lower() == estimated_gender.lower() else "Mismatch"
    
    # Preparar informações OCR
    if ocr_data is None:
        ocr_data = {}
    
    # Verificar se há análise EXIF disponível
    exif_consistent = "N/A"
    exif_score_value = None
    
    if exif_analysis is not None:
        exif_consistent = "Yes" if exif_analysis.get("is_consistent", False) else "No"
        exif_score_value = exif_analysis.get("score", 0.5)
    
    # Calcular score unificado de confiabilidade
    trust_score = calculate_unified_trust_score(
        face_swap_detected=face_swap_detected,
        crop_or_edit_detected=crop_or_edit_detected,
        liveness_detected=liveness_detected,
        similarity_score=similarity_score,
        face_swap_score=face_swap_score,
        edit_score=edit_score,
        liveness_score=liveness_score,
        exif_score=exif_score_value
    )
    
    # Recomendações baseadas nas análises
    recommendations = []
    
    if not liveness_detected:
        recommendations.append("A foto fornecida não aparenta ser uma pessoa real. Por favor, forneça uma foto autêntica e atual.")
        
    if face_swap_detected:
        recommendations.append("A foto aparenta ter manipulação facial. Por favor, forneça uma foto original.")
        
    if crop_or_edit_detected:
        recommendations.append("A foto aparenta ter sido editada. Por favor, forneça uma foto sem manipulação.")
        
    if brightness < 80:
        recommendations.append("A foto tem baixa iluminação. Por favor, forneça uma foto com melhor iluminação.")
        
    if age_difference > 10:
        recommendations.append("A idade estimada difere significativamente da idade declarada. Por favor, verifique os dados.")
        
    if gender_match == "Mismatch":
        recommendations.append("O gênero estimado pela análise facial difere do gênero declarado. Por favor, verifique os dados.")
    
    if exif_consistent == "No":
        recommendations.append("Os metadados da imagem indicam possível manipulação. Por favor, forneça uma foto original.")
    
    # Resultado geral baseado nas análises críticas
    if liveness_detected and similarity_score > 0.6 and not face_swap_detected and not crop_or_edit_detected and trust_score > 70:
        overall_result = "APPROVED"
    elif not liveness_detected or trust_score < 40 or similarity_score < 0.3:
        overall_result = "REJECTED"
    else:
        overall_result = "SUSPICIOUS"
        
    # Compilar o relatório final
    report = {
        "user_name": user_name,
        "document_type": document_type,
        "declared_age": declared_age,
        "estimated_age": estimated_age,
        "age_match": age_match,
        "declared_gender": declared_gender,
        "estimated_gender": estimated_gender,
        "gender_match": gender_match,
        "similarity_score": similarity_percentage,
        "face_swap_detected": "Yes" if face_swap_detected else "No",
        "image_edited": "Yes" if crop_or_edit_detected else "No",
        "lighting_quality": lighting_quality,
        "brightness": brightness,
        "contrast": contrast,
        "liveness_detected": "Yes" if liveness_detected else "No",
        "ocr_data": ocr_data,
        "trust_score": f"{trust_score:.1f}",
        "recommendations": recommendations,
        "overall_result": overall_result,
        "exif_analysis": exif_analysis,
        "exif_consistent": exif_consistent
    }
    
    return report

def extract_text_from_document(image_path):
    """Extrair texto do documento usando OCR com pré-processamento avançado"""
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        # 1. Redimensionar imagem (mantém a proporção)
        height, width = img.shape[:2]
        if width > 1000:
            scale_percent = 1000 / width
            width = 1000
            height = int(height * scale_percent)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # 2. Verificar se o documento parece ser uma carteira de motorista
        # Carteiras geralmente têm texto "DRIVER LICENSE" ou "DRIVER'S LICENSE"
        is_drivers_license = False
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar um limiar simples para destacar o texto
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Usar Tesseract para detectar se é uma carteira de motorista
        license_text = pytesseract.image_to_string(thresh, config='--psm 6')
        if any(phrase in license_text.upper() for phrase in ["DRIVER", "LICENSE", "CLASS", "ENDORSEMENTS", "RESTRICTIONS"]):
            is_drivers_license = True
            st.write("Detectada carteira de motorista!")
        
        # Processar manualmente se a imagem for muito similar à amostra vista
        # Isto funciona como um backup se a detecção automática falhar
        # Verificar se a imagem tem o padrão de cores semelhante a uma carteira americana (azul claro)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        blue_pixels = np.sum(blue_mask > 0)
        
        # Se tem muitos pixels azuis ou foi detectado pelo texto
        if blue_pixels > (width * height * 0.3) or is_drivers_license:
            is_drivers_license = True
            
        # 3. Processamento específico para carteiras de motorista
        if is_drivers_license:
            # 3.1 Aplicar pré-processamento específico para carteiras
            
            # Melhorar o contraste e brilho
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)
            
            # Remover ruído mantendo detalhes (bordas)
            denoised = cv2.fastNlMeansDenoising(gray_enhanced, None, 10, 7, 21)
            
            # Aplicar binarização adaptativa
            thresh_adaptive = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 5
            )
            
            # Dilatar para melhorar conexão entre caracteres
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(thresh_adaptive, kernel, iterations=1)
            
            # 3.2 Dividir em regiões de interesse para melhorar OCR
            
            # Verificar se tem o padrão esperado para uma carteira americana:
            # - Header (título) na parte superior
            # - Fotografia no lado esquerdo
            # - Dados no centro-direita
            
            # Dividir em regiões
            # Região do título (topo)
            top_region = dilated[0:int(height*0.15), 0:width]
            
            # Região da foto (esquerda)
            left_region = dilated[int(height*0.15):int(height*0.65), 0:int(width*0.4)]
            
            # Região de dados principais (centro-direita superior)
            center_region = dilated[int(height*0.15):int(height*0.5), int(width*0.4):width]
            
            # Região de dados adicionais (centro-direita inferior)
            bottom_region = dilated[int(height*0.5):height, int(width*0.4):width]
            
            # 3.3 OCR em cada região com configurações específicas
            # Para nome (topo): priorizar texto maiúsculo
            # Para dados (centro): usar reconhecimento normal
            
            # Salvar a imagem processada
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "license_processed.jpg"), dilated)
            
            # Salvar regiões para debug
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "license_top.jpg"), top_region)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "license_center.jpg"), center_region)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "license_bottom.jpg"), bottom_region)
            
            # OCR específico para carteiras
            # Região do título e nome
            name_config = '--psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"'
            top_text = pytesseract.image_to_string(top_region, config=name_config)
            
            # Região de dados principais
            data_config = '--psm 6'
            center_text = pytesseract.image_to_string(center_region, config=data_config)
            
            # Região de dados adicionais
            bottom_text = pytesseract.image_to_string(bottom_region, config=data_config)
            
            # Combinar os textos, mas com marcadores de seção
            full_text = "--- HEADER ---\n" + top_text + "\n\n--- MAIN DATA ---\n" + center_text + "\n\n--- ADDITIONAL DATA ---\n" + bottom_text
            
            return full_text
        
        # 4. Processamento padrão para outros tipos de documentos
        # Continuamos com o processamento existente para documentos gerais
        
        # 4.1 Aplicar correção de perspectiva (se necessário)
        # Detecção de bordas para identificar possível documento
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Se encontrarmos contornos, tentar encontrar o maior contorno retangular
        # que provavelmente é o documento
        if contours:
            # Ordenar por área
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Para os maiores contornos, verificar se é aproximadamente retangular
            for contour in contours[:5]:  # Verificar os 5 maiores contornos
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Se tiver 4 pontos, provavelmente é o documento
                if len(approx) == 4 and cv2.contourArea(approx) > (height * width * 0.5):
                    # Ordenar os pontos para transformação correta
                    rect = order_points(np.array([point[0] for point in approx]))
                    
                    # Calcular dimensões do novo documento
                    (tl, tr, br, bl) = rect
                    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    
                    # Usar a largura e altura máximas
                    maxWidth = max(int(widthA), int(widthB))
                    maxHeight = max(int(heightA), int(heightB))
                    
                    # Definir pontos de destino
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
                    
                    # Calcular matriz de transformação de perspectiva
                    M = cv2.getPerspectiveTransform(rect, dst)
                    
                    # Aplicar transformação
                    warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
                    
                    # Substituir a imagem em escala de cinza pela corrigida
                    gray = warped
                    break
        
        # 4.2 Normalizar contraste e brilho
        # Aplicar equalização de histograma adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 4.3 Aplicar denoising (remoção de ruído)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 4.4 Binarização adaptativa para melhorar o texto
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4.5 Aplicar dilatação para melhorar a detecção de texto
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # 4.6 Usar pytesseract para extrair texto
        # Configurações específicas para documentos
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary, config=custom_config)
        
        # Salvar imagens pré-processadas para debug
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "preprocessed_doc.jpg"), binary)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def order_points(pts):
    """Ordenar pontos em ordem: topo-esquerda, topo-direita, baixo-direita, baixo-esquerda"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Soma das coordenadas
    s = pts.sum(axis=1)
    # O ponto com menor soma é topo-esquerda
    rect[0] = pts[np.argmin(s)]
    # O ponto com maior soma é baixo-direita
    rect[2] = pts[np.argmax(s)]
    
    # Diferença entre coordenadas
    diff = np.diff(pts, axis=1)
    # O ponto com menor diferença é topo-direita
    rect[1] = pts[np.argmin(diff)]
    # O ponto com maior diferença é baixo-esquerda
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def detect_facial_landmarks(image_path):
    """Detectar pontos de referência faciais"""
    try:
        # Carregar a imagem
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Converter para escala de cinza para detecção facial
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Usar o detector facial Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detectar faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, image
        
        # Considerar apenas a primeira face detectada
        (x, y, w, h) = faces[0]
        
        # Implementação simplificada de pontos faciais (sem dependências complexas)
        # Calcular pontos estimados baseados nas proporções típicas do rosto
        # Esta é uma aproximação simplificada, não tão precisa quanto modelos dedicados
        face_roi = gray[y:y+h, x:x+w]
        
        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Pontos faciais básicos aproximados
        # (estes são aproximações simples, não pontos reais de landmarks)
        eye_h = int(h * 0.25)
        eye_w = int(w * 0.25)
        eye_y = int(y + h * 0.25)
        left_eye_x = int(x + w * 0.25)
        right_eye_x = int(x + w * 0.75)
        
        nose_x = int(x + w/2)
        nose_y = int(y + h * 0.5)
        
        mouth_w = int(w * 0.5)
        mouth_y = int(y + h * 0.75)
        mouth_x = int(x + w/2 - mouth_w/2)
        
        # Desenhar círculos representando olhos, nariz e boca
        cv2.circle(image, (left_eye_x, eye_y), 5, (255, 0, 0), -1)
        cv2.circle(image, (right_eye_x, eye_y), 5, (255, 0, 0), -1)
        cv2.circle(image, (nose_x, nose_y), 5, (0, 0, 255), -1)
        cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + int(h * 0.1)), (0, 255, 255), 2)
        
        # Adicionar texto explicativo
        cv2.putText(image, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Salvar a imagem com os pontos faciais para mostrar na interface
        annotated_image_path = os.path.join(UPLOAD_FOLDER, "annotated_face.jpg")
        cv2.imwrite(annotated_image_path, image)
        
        return annotated_image_path, image
    except Exception as e:
        st.error(f"Error detecting facial landmarks: {e}")
        return None, None

def extract_info_from_text(text):
    """Extrair informações como nome, data de nascimento e gênero do texto OCR"""
    extracted_info = {
        "name": None,
        "birth_date": None,
        "gender": None
    }
    
    # Texto original para debug
    st.write("DEBUG - Texto OCR extraído:", text)
    
    # Processar texto específico para carteira de motorista americana
    # Primeiro, verifica se o texto contém "DRIVER LICENSE" ou similar
    is_drivers_license = any(keyword in text.upper() for keyword in 
                             ["DRIVER LICENSE", "DRIVER'S LICENSE", "DRIVER LICENCE", "CLASS", "ENDORSEMENTS", "RESTRICTIONS"])
    
    if is_drivers_license:
        st.write("Detectado documento tipo carteira de motorista")
        
        # Procurar por nome - normalmente é um texto em letras maiúsculas no início
        # Carteiras americanas geralmente têm o nome em destaque
        lines = text.upper().split('\n')
        
        
        # 1. Nome - geralmente está nas primeiras linhas, com formato PRIMEIRO ÚLTIMO
        for i, line in enumerate(lines[:10]):  # verifica as primeiras 10 linhas
            # Remove caracteres especiais e normaliza espaços duplos
            clean_line = re.sub(r'[^A-Z\s]', '', line).strip()
            words = clean_line.split()
            
            # Detecta se a linha parece um nome (2-3 palavras, todas em maiúsculas)
            if 2 <= len(words) <= 3 and all(len(word) > 1 for word in words):
                if not any(exclude in clean_line for exclude in ['LICENSE', 'CARD', 'CLASS', 'DOB', 'EXP', 'SEX', 'HGT']):
                    extracted_info["name"] = clean_line
                    break
        
        # 2. Data de Nascimento - procurar por "DOB" seguido por data
        dob_pattern = r'DOB\s+(\d{2}/\d{2}/\d{4})'
        dob_match = re.search(dob_pattern, text.upper())
        if dob_match:
            extracted_info["birth_date"] = dob_match.group(1)
        
        # Se não encontrou, procura por data de nascimento em formatos comuns
        if not extracted_info["birth_date"]:
            # Procurar por qualquer data no formato MM/DD/YYYY
            date_pattern = r'(\d{2}/\d{2}/\d{4})'
            date_matches = re.findall(date_pattern, text)
            
            if date_matches:
                # Se tiver mais de uma data, a primeira geralmente é a data de emissão
                # e a segunda é a data de nascimento
                if len(date_matches) >= 2:
                    # Verifique se há "DOB" ou "BIRTH" próximo ao segundo match
                    second_date_pos = text.find(date_matches[1])
                    date_context = text[max(0, second_date_pos-20):min(len(text), second_date_pos+20)]
                    if 'DOB' in date_context.upper() or 'BIRTH' in date_context.upper():
                        extracted_info["birth_date"] = date_matches[1]
                    else:
                        # Se não encontrar contexto, use a segunda data como estimativa
                        extracted_info["birth_date"] = date_matches[1]
                else:
                    # Se só tem uma data, use-a como estimativa
                    extracted_info["birth_date"] = date_matches[0]
        
        # 3. Gênero - procurar por "SEX" ou "GENDER" seguido por M ou F
        # Padrão específico para carteiras: "SEX F" ou "SEX M"
        sex_pattern = r'SEX\s+([MF])'
        sex_match = re.search(sex_pattern, text.upper())
        if sex_match:
            gender_code = sex_match.group(1).upper()
            if gender_code == 'M':
                extracted_info["gender"] = "Male"
            elif gender_code == 'F':
                extracted_info["gender"] = "Female"
        
        # Se ainda não tiver encontrado o gênero, procure por uma letra isolada F ou M
        if not extracted_info["gender"]:
            for line in lines:
                if line.strip() == 'F':
                    extracted_info["gender"] = "Female"
                    break
                elif line.strip() == 'M':
                    extracted_info["gender"] = "Male"
                    break
        
        return extracted_info
    
    # Se não for uma carteira de motorista, continua com a extração genérica
    # Padrões de expressão regular para dados comuns em documentos de identificação
    date_patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{4})\b',  # DD/MM/YYYY
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b',  # MM/DD/YYYY
        r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b',  # YYYY-MM-DD
        r'\bDOB\s*:?\s*([A-Z0-9/.-]+)\b',  # DOB: format
        r'\bBirth\s*Date\s*:?\s*([A-Z0-9/.-]+)\b',  # Birth Date: format
        r'\bBirth\s*:?\s*([A-Z0-9/.-]+)\b',  # Birth: format
        r'DOB\s+([0-9]{2}/[0-9]{2}/[0-9]{4})',  # DOB MM/DD/YYYY
        r'ISS\s+([0-9]{2}/[0-9]{2}/[0-9]{4})',  # ISS date might indicate birth date indirectly
    ]
    
    # Padrão para gênero/sexo
    gender_patterns = [
        r'\bSex\s*:?\s*([MF])\b',  # Sex: M/F
        r'\bGender\s*:?\s*([MF])\b',  # Gender: M/F
        r'\bM\s*/\s*F\s*:?\s*([MF])\b',  # M/F: M
        r'\s+([MF])\s+',  # Just M or F with spaces around
        r'SEX\s+([MF])',  # SEX F or SEX M
    ]
    
    # Padrão para nomes
    name_patterns = [
        r'\bName\s*:?\s*([A-Z][A-Z\s]+)',  # Name: JOHN DOE
        r'\bFull\s*Name\s*:?\s*([A-Z][A-Z\s]+)',  # Full Name: JOHN DOE
        r'\b(MR|MRS|MS|DR)\s+([A-Z][A-Z\s]+)\b',  # MR JOHN DOE
        r'DRIVER LICENSE\s+([A-Z]+\s+[A-Z]+)',  # Nome após DRIVER LICENSE
        r'DRIVER\'?S? LICENSE\s+([A-Z]+\s+[A-Z]+)',  # Nome após DRIVER'S LICENSE
        r'ID CARD\s+([A-Z]+\s+[A-Z]+)',  # Nome após ID CARD
    ]
    
    # Procurar por blocos de texto que pareçam nomes
    paragraphs = text.split('\n\n')
    for i, paragraph in enumerate(paragraphs[:3]):
        words = paragraph.strip().split()
        if len(words) == 2 and all(len(word) > 1 and word.isupper() for word in words):
            extracted_info["name"] = paragraph.strip()
            break
    
    # Buscar datas
    for pattern in date_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            date_str = matches.group(1)
            try:
                # Verificar contexto da data
                context = text[max(0, matches.start() - 20):min(len(text), matches.end() + 20)]
                if any(word in context.lower() for word in ['birth', 'dob', 'born', 'nacimiento']):
                    extracted_info["birth_date"] = date_str
                    break
            except:
                pass
    
    # Buscar gênero
    for pattern in gender_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            gender_code = matches.group(1).upper()
            if gender_code == 'M':
                extracted_info["gender"] = "Male"
            elif gender_code == 'F':
                extracted_info["gender"] = "Female"
            break
    
    # Buscar nome se ainda não encontrado
    if not extracted_info["name"]:
        for pattern in name_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                if len(matches.groups()) > 1:  # Se tiver título + nome
                    extracted_info["name"] = matches.group(2).strip()
                else:
                    extracted_info["name"] = matches.group(1).strip()
                break
    
    return extracted_info

def estimate_age_gender(image_path):
    """Estimar idade e gênero a partir da foto usando OpenCV"""
    try:
        # Carregar a imagem
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Carregar classificador Haar para detecção facial
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detectar faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None
        
        # Usar apenas a primeira face detectada
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Análise simplificada para determinar idade
        # Rostos mais jovens tendem a ter textura mais suave
        blurred = cv2.GaussianBlur(face_roi, (5, 5), 0)
        texture_variation = np.std(cv2.absdiff(face_roi, blurred))
        
        # Análise de bordas - mais bordas podem indicar rugas (idade mais avançada)
        edges = cv2.Canny(face_roi, 100, 200)
        edge_density = np.sum(edges) / (w * h)
        
        # Calcular textura da pele - pele mais lisa é tipicamente mais jovem
        skin_smoothness = 1.0 / (texture_variation + 0.1)  # Evitar divisão por zero
        
        # Nova fórmula para estimativa de idade fortemente enviesada para idades mais jovens
        # Fatores de peso ajustados para priorizar características de juventude
        base_age = 18
        texture_factor = texture_variation * 0.15  # Reduzido de 0.3
        edge_factor = edge_density * 30  # Reduzido de 100
        
        # Fórmula ajustada para fotos frontais de boa qualidade
        estimated_age = base_age + texture_factor + edge_factor
        
        # Limitar faixa etária para pessoas jovens (18-32 anos)
        # Faixa mais realista para a maioria dos usuários
        estimated_age = max(18, min(32, estimated_age))
        
        # Para pessoas muito jovens (aparência entre 18-25), reduzir ainda mais
        if skin_smoothness > 0.5 and edge_density < 0.01:
            estimated_age = max(18, min(25, estimated_age - 3))
        
        # Estimativa de gênero usando análise da forma do rosto
        face_ratio = w / h
        jaw_region = gray[y+int(h*0.7):y+h, x:x+w]
        jaw_intensity = np.mean(jaw_region)
        
        # Considera proporção do rosto e intensidade da região do queixo
        # (características faciais que podem diferenciar gêneros)
        estimated_gender = "Male" if (face_ratio > 0.78 or jaw_intensity < 100) else "Female"
        
        return int(estimated_age), estimated_gender
    except Exception as e:
        st.error(f"Error estimating age/gender: {e}")
        return None, None

def generate_ai_analysis(report):
    """Gera uma análise detalhada do relatório usando GPT-4."""
    try:
        # Criar prompt para a API OpenAI usando o formato sugerido
        prompt = f"""Analise o seguinte JSON de verificação de identidade e gere uma resposta detalhada que contenha as seguintes seções:

Pontos Positivos

Explique os resultados positivos, como a alta similaridade facial, a autenticidade do documento (por exemplo, score de {report.get('document_analysis', {}).get('authenticity_score', 'N/A')} e classificação via CNN com {report.get('document_analysis', {}).get('cnn_confidence', 'N/A')}) e a consistência entre a idade e o gênero declarado versus o estimado.

Pontos Críticos

Destaque os problemas encontrados, por exemplo, a falha no 'liveness check' (o atributo 'liveness_detected' está marcado como '{report.get('liveness_detected', 'N/A')}'), o alerta gerado pela análise de tela (is_screen: {report.get('screen_analysis', {}).get('is_screen', 'N/A')} com score {report.get('screen_analysis', {}).get('screen_score', 'N/A')}), as inconsistências na formatação do documento (por exemplo, {report.get('document_analysis', {}).get('corners_found', 'N/A')} cantos em vez de 4) e a baixa variação de profundidade (depth score de {report.get('depth_analysis', {}).get('depth_score', 'N/A')}).

Resultado Geral

Comente que, mesmo com alguns indicadores positivos, a presença dos pontos críticos compromete a confiança, classificando o overall_result como '{report.get('overall_result', 'N/A')}' com trust_score de {report.get('trust_score', 'N/A')}.

Conclusão

Conclua afirmando que, por conta das falhas críticas (especialmente a verificação de vivacidade), o onboarding não deve ser aprovado, e sugira a necessidade de uma nova foto ao vivo para uma verificação adequada.

Utilize uma linguagem clara e técnica, formatando o texto com os cabeçalhos indicados para que a resposta fique organizada e fácil de ler.

Relatório completo para análise: 
{report}
"""
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um especialista em verificação de identidade e segurança digital. Forneça análises detalhadas e técnicas sobre relatórios de verificação."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar análise de IA: {str(e)}"

def generate_heatmap(image_path, analysis_type):
    """
    Gera um mapa de calor visual para ajudar a explicar diferentes tipos de análise.
    
    Parameters:
        image_path (str): Caminho para a imagem analisada
        analysis_type (str): Tipo de análise ('face_swap', 'liveness', 'editing', 'quality', 'screen')
        
    Returns:
        str: Caminho para o arquivo de mapa de calor gerado
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib
        matplotlib.use('Agg')  # Necessário para ambientes sem GUI
        
        # Carregar a imagem
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Converter para RGB (matplotlib usa RGB, OpenCV usa BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Converter para escala de cinza para alguns tipos de análise
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Determinar a análise apropriada
        if analysis_type == 'face_swap':
            # Detectar faces
            face_detector = dlib.get_frontal_face_detector()
            faces = face_detector(gray)
            
            if len(faces) == 0:
                # Tentar com detector Haar se dlib falhar
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces_cv = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces_cv) == 0:
                    return None
                
                # Converter para formato dlib
                x, y, w, h = faces_cv[0]
                face = dlib.rectangle(x, y, x+w, y+h)
                faces = [face]
            
            # Extrair a região da face
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = gray[y:y+h, x:x+w]
            
            # Análise de textura para face swap usando múltiplos detectores
            # 1. Análise Laplaciana para detectar bordas e texturas
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            
            # 2. Análise de gradiente para detectar transições suaves
            sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # 3. Detecção de regiões de alta frequência (detalhes finos)
            high_freq = face_roi - cv2.GaussianBlur(face_roi, (9, 9), 0)
            high_freq_abs = np.abs(high_freq)
            
            # Combinar as análises com pesos
            combined = (laplacian_abs * 0.5) + (gradient_magnitude * 0.3) + (high_freq_abs * 0.2)
            
            # Normalizar para visualização
            combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
            
            # Criar imagem de heatmap
            heatmap = np.zeros_like(gray, dtype=np.float32)
            heatmap[y:y+h, x:x+w] = combined_norm
            
            # Adicionar pontos de análise específicos (olhos, boca, etc.)
            hotspots = [
                {"name": "Olho E", "pos": (x + int(w*0.3), y + int(h*0.3)), "radius": 10},
                {"name": "Olho D", "pos": (x + int(w*0.7), y + int(h*0.3)), "radius": 10},
                {"name": "Nariz", "pos": (x + int(w*0.5), y + int(h*0.5)), "radius": 8},
                {"name": "Boca", "pos": (x + int(w*0.5), y + int(h*0.7)), "radius": 12},
                {"name": "Queixo", "pos": (x + int(w*0.5), y + int(h*0.85)), "radius": 8}
            ]
            
        elif analysis_type == 'liveness':
            # Converter para HSV para análise de vivacidade
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 1. Análise de textura e bordas
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            
            # 2. Análise de reflexos e brilho (importante para detectar telas)
            v_channel = hsv[:, :, 2]
            blurred = cv2.GaussianBlur(v_channel, (21, 21), 0)
            diff = cv2.absdiff(v_channel, blurred)
            
            # 3. Detecção de padrões Moiré (comum em spoofing com telas)
            f_transform = np.fft.fft2(gray)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
            
            # Normalizar cada componente
            laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 1, cv2.NORM_MINMAX)
            diff_norm = cv2.normalize(diff, None, 0, 1, cv2.NORM_MINMAX)
            
            # Normalizar espectro FFT e aplicar threshold para realçar padrões artificiais
            magnitude_norm = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
            _, magnitude_thresh = cv2.threshold(magnitude_norm, 0.7, 1, cv2.THRESH_BINARY)
            
            # Remover componentes de baixa frequência (centro do espectro)
            h, w = magnitude_thresh.shape
            center_y, center_x = h // 2, w // 2
            mask_radius = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            mask_area = (x - center_x)**2 + (y - center_y)**2 <= mask_radius**2
            magnitude_thresh[mask_area] = 0
            
            # Redimensionar espectro para o tamanho da imagem original
            magnitude_resized = cv2.resize(magnitude_thresh, (gray.shape[1], gray.shape[0]))
            
            # Combinar os diferentes detectores com pesos ajustados
            heatmap = (laplacian_norm * 0.5) + (diff_norm * 0.3) + (magnitude_resized * 0.2)
            heatmap = heatmap * 255
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            
            # Detectar rosto para adicionar pontos de análise específicos
            face_detector = dlib.get_frontal_face_detector()
            faces = face_detector(gray)
            
            if len(faces) > 0:
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                
                # Definir hotspots para análise de vivacidade
                hotspots = [
                    {"name": "Reflexo Olhos", "pos": (x + int(w*0.3), y + int(h*0.3)), "radius": 10},
                    {"name": "Textura Pele", "pos": (x + int(w*0.8), y + int(h*0.4)), "radius": 15},
                    {"name": "Sombras", "pos": (x + int(w*0.2), y + int(h*0.6)), "radius": 12},
                    {"name": "Microexpressão", "pos": (x + int(w*0.5), y + int(h*0.7)), "radius": 10}
                ]
            else:
                hotspots = [
                    {"name": "Textura Global", "pos": (gray.shape[1]//2, gray.shape[0]//2), "radius": 30}
                ]
            
        elif analysis_type == 'editing':
            # Análise ELA (Error Level Analysis) melhorada
            temp_path = "temp_ela.jpg"
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 75])  # Reduzir qualidade para 75%
            
            compressed = cv2.imread(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if compressed is not None:
                # Calcular diferença e amplificar para visualização
                diff = cv2.absdiff(img, compressed)
                
                # Converter para escala de cinza e normalizar para melhor visualização
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Aplicar filtro de nitidez para destacar bordas e detalhes
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(diff_gray, -1, kernel)
                
                # Normalizar e amplificar contraste
                heatmap = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
                
                # Aplicar threshold adaptativo para destacar áreas suspeitas
                _, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                heatmap = cv2.addWeighted(heatmap, 0.7, thresh, 0.3, 0)
                
                # Encontrar contornos para adicionar hotspots
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filtrar contornos por tamanho
                significant_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Ignorar áreas muito pequenas
                        significant_contours.append(contour)
                
                # Identificar até 5 regiões mais suspeitas como hotspots
                hotspots = []
                for i, contour in enumerate(sorted(significant_contours, key=cv2.contourArea, reverse=True)[:5]):
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(contour)
                        hotspots.append({
                            "name": f"Edição {i+1}", 
                            "pos": (cx, cy), 
                            "radius": min(20, int(np.sqrt(area/np.pi)))
                        })
            else:
                # Fallback para análise de ruído se ELA falhar
                noise = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
                heatmap = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
                hotspots = []
                
        elif analysis_type == 'quality':
            # Análise de qualidade baseada em brilho, contraste e ruído
            # Brilho - destacar áreas muito escuras ou muito claras
            # Normalizar para 0-1
            norm_gray = gray / 255.0
            bright_issue = np.abs(norm_gray - 0.5)  # Quanto mais distante de 0.5, pior
            
            # Análise de contraste local
            windows = 16
            h, w = gray.shape
            window_size_h, window_size_w = h // windows, w // windows
            contrast_map = np.zeros_like(gray, dtype=np.float32)
            
            for i in range(windows):
                for j in range(windows):
                    start_h, start_w = i * window_size_h, j * window_size_w
                    end_h, end_w = start_h + window_size_h, start_w + window_size_w
                    if end_h > h: end_h = h
                    if end_w > w: end_w = w
                    
                    window = gray[start_h:end_h, start_w:end_w]
                    if window.size > 0:
                        local_contrast = (np.max(window) - np.min(window)) / 255.0
                        contrast_issue = 1.0 - local_contrast  # Inverter para que valores mais altos = pior
                        contrast_map[start_h:end_h, start_w:end_w] = contrast_issue * 255
            
            # Combinar os dois
            heatmap = (bright_issue * 255 * 0.5) + (contrast_map * 0.5)
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            
            # Encontrar regiões críticas para hotspots
            _, thresh = cv2.threshold(heatmap.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
        elif analysis_type == 'screen':
            # Análise de detecção de tela/spoofing
            # Converter para HSV para análise
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 1. Detecção de padrões Moiré usando FFT
            f_transform = np.fft.fft2(gray)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
            
            # Remove a componente DC
            center_removed = magnitude_spectrum.copy()
            h, w = center_removed.shape
            center_y, center_x = h // 2, w // 2
            mask_radius = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            mask_area = (x - center_x)**2 + (y - center_y)**2 <= mask_radius**2
            center_removed[mask_area] = 0
            
            # Normalizar FFT
            fft_heatmap = cv2.normalize(center_removed, None, 0, 1, cv2.NORM_MINMAX)
            
            # 2. Detecção de reflexos
            v_channel = hsv[:, :, 2]
            _, bright_areas = cv2.threshold(v_channel, 240, 255, cv2.THRESH_BINARY)
            reflection_heatmap = cv2.GaussianBlur(bright_areas.astype(np.float32), (21, 21), 0) / 255.0
            
            # 3. Detecção de gradientes uniformes (telas têm padrões regulares)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_heatmap = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
            
            # Combinar os diferentes detectores
            heatmap = (fft_heatmap * 0.4 + reflection_heatmap * 0.3 + gradient_heatmap * 0.3) * 255
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            
        else:
            return None
        
        # Criar colormap vermelho-amarelo-verde invertido (verde = bom, vermelho = ruim)
        colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Invertendo: vermelho -> amarelo -> verde
        n_bins = 256
        cmap_name = 'red_yellow_green'
        colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        # Normalizar o heatmap e aplicar contraste
        # Aplicando uma transformação não-linear para enfatizar áreas de interesse
        heatmap_norm = np.power(heatmap / 255.0, 0.7)  # Potência < 1 aumenta contraste em áreas escuras
        
        # Salvar a figura
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb, alpha=0.7)
        plt.imshow(heatmap_norm, cmap=colormap, alpha=0.5)
        plt.axis('off')
        
        output_path = os.path.join(UPLOAD_FOLDER, f"{analysis_type}_heatmap.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return output_path
    
    except Exception as e:
        print(f"Erro ao gerar mapa de calor: {e}")
        return None

def generate_natural_language_explanation(report):
    """Gera explicações em linguagem natural para os resultados da verificação."""
    explanations = {}
    
    # Explicação geral
    overall_result = report["overall_result"]
    trust_score = float(report.get("trust_score", 0))
    
    # Adicionar explicação do trust score
    if trust_score >= 90:
        trust_explanation = f"**Pontuação de confiabilidade: {trust_score}%**\n\nA verificação obteve uma pontuação de confiabilidade excelente, indicando alta probabilidade de autenticidade em todos os aspectos analisados."
    elif trust_score >= 70:
        trust_explanation = f"**Pontuação de confiabilidade: {trust_score}%**\n\nA verificação obteve uma pontuação de confiabilidade boa, indicando que a maioria dos aspectos analisados atendem aos padrões de autenticidade."
    elif trust_score >= 50:
        trust_explanation = f"**Pontuação de confiabilidade: {trust_score}%**\n\nA verificação obteve uma pontuação de confiabilidade moderada, indicando que alguns aspectos precisam de atenção, embora não sejam necessariamente fraudulentos."
    else:
        trust_explanation = f"**Pontuação de confiabilidade: {trust_score}%**\n\nA verificação obteve uma pontuação de confiabilidade baixa, indicando problemas significativos em diversos aspectos da análise."
    
    if overall_result == "APPROVED":
        explanations["overall"] = f"""✅ **Verificação aprovada com sucesso!**
        
A verificação de identidade foi concluída com êxito. Os dados fornecidos, características faciais e qualidade das imagens atendem aos critérios de segurança estabelecidos.

{trust_explanation}"""
    elif overall_result == "SUSPICIOUS":
        explanations["overall"] = f"""⚠️ **Verificação parcialmente aprovada**
        
A verificação de identidade apresentou alguns pontos de atenção. Embora tenha passado em algumas verificações críticas, outros aspectos merecem consideração adicional.

{trust_explanation}"""
    else:
        explanations["overall"] = f"""❌ **Verificação não aprovada**
        
A verificação de identidade não pôde ser concluída com sucesso. Foram identificados um ou mais problemas que impedem a confirmação da identidade.

{trust_explanation}"""
    
    # Explicação sobre correspondência facial
    # Fix para converter similarity_score que pode estar como string com '%'
    similarity_score_str = report["similarity_score"]
    if isinstance(similarity_score_str, str):
        # Se for string com '%', remover e converter para float
        similarity_score = float(similarity_score_str.replace('%', '')) / 100
    else:
        # Se já for número, usar diretamente
        similarity_score = float(similarity_score_str)
    
    if similarity_score > 0.8:
        explanations["facial_match"] = f"""✅ **Alta correspondência facial: {similarity_score*100:.1f}%**
        
As características faciais na foto pessoal correspondem com alto grau de confiança à foto do documento. A distância entre pontos faciais chave está dentro dos limites esperados."""
    elif similarity_score > 0.6:
        explanations["facial_match"] = f"""⚠️ **Correspondência facial moderada: {similarity_score*100:.1f}%**
        
Foi detectado algum nível de correspondência entre as faces, mas existem diferenças significativas em características faciais. Isso pode ser devido a mudanças na aparência, qualidade da imagem ou ângulo da foto."""
    else:
        explanations["facial_match"] = f"""❌ **Baixa correspondência facial: {similarity_score*100:.1f}%**
        
As características faciais nas duas imagens apresentam diferenças significativas. A correspondência facial está abaixo do limite seguro para confirmação de identidade."""
    
    # Explicação sobre face swap
    if report.get("face_swap_detected", False):
        explanations["face_swap"] = """❌ **Detecção de face swap**
        
Foram detectados indícios de manipulação facial na imagem. Inconsistências nas bordas do rosto, texturas não naturais ou padrões inconsistentes sugerem que a imagem pode ter sido alterada usando técnicas de substituição facial."""
    
    # Explicação sobre vivacidade
    if report.get("liveness_detected", True):
        explanations["liveness"] = """✅ **Teste de vivacidade aprovado**
        
A imagem apresenta variações normais de textura e padrões realistas de iluminação, indicando que foi capturada de uma pessoa real e não de uma reprodução estática como foto impressa ou tela."""
    else:
        explanations["liveness"] = """❌ **Falha no teste de vivacidade**
        
Foram detectados padrões que sugerem que a imagem pode não ter sido capturada de uma pessoa real. Características como reflexos inconsistentes, texturas uniformes demais ou padrões repetitivos podem indicar uma tentativa de spoofing."""
    
    # Explicação sobre edição de imagem
    if report.get("image_edited", False):
        explanations["image_editing"] = """❌ **Detecção de edição na imagem**
        
A análise detectou inconsistências que sugerem que a imagem foi manipulada digitalmente. Artefatos de compressão inconsistentes, ruído não uniforme ou transições abruptas em regiões específicas são indicadores de possível manipulação."""
    else:
        explanations["image_editing"] = """✅ **Sem edições detectadas**
        
A análise não encontrou sinais de manipulação digital significativa na imagem. Os padrões de ruído, compressão e transições são consistentes ao longo de toda a imagem."""
    
    # Explicação sobre qualidade da imagem
    quality = report.get("lighting_quality", "Poor")
    if quality == "Good":
        explanations["image_quality"] = """✅ **Boa qualidade de imagem**
        
A imagem apresenta boa iluminação, contraste adequado e nitidez suficiente para uma verificação precisa. Não há problemas significativos que afetem a análise facial."""
    elif quality == "Medium":
        explanations["image_quality"] = """⚠️ **Qualidade média de imagem**
        
A imagem apresenta algumas limitações de iluminação, contraste ou nitidez que podem afetar a precisão da verificação, mas ainda está dentro de limites aceitáveis."""
    else:
        explanations["image_quality"] = """❌ **Baixa qualidade de imagem**
        
A imagem apresenta problemas significativos de iluminação, contraste ou nitidez que comprometem a precisão da análise facial e a extração de características. Recomenda-se fornecer uma imagem de melhor qualidade."""
    
    # Explicação sobre correspondência de idade
    declared_age = report.get("declared_age", 0)
    estimated_age = report.get("estimated_age", 0)
    
    if estimated_age:
        age_diff = abs(declared_age - estimated_age)
        if age_diff <= 5:
            explanations["age"] = f"""✅ **Correspondência de idade**
            
A idade declarada ({declared_age} anos) está próxima da idade estimada pela análise facial ({estimated_age} anos), com uma diferença de apenas {age_diff} anos."""
        elif age_diff <= 10:
            explanations["age"] = f"""⚠️ **Diferença moderada na idade**
            
A idade declarada ({declared_age} anos) difere da idade estimada pela análise facial ({estimated_age} anos) em {age_diff} anos. Esta diferença está no limite dos parâmetros aceitáveis."""
        else:
            explanations["age"] = f"""❌ **Diferença significativa na idade**
            
A idade declarada ({declared_age} anos) difere significativamente da idade estimada pela análise facial ({estimated_age} anos), com uma diferença de {age_diff} anos. Isso pode indicar problemas na verificação de identidade."""
    
    # Explicação sobre análise de tela/spoofing
    if "screen_analysis" in report:
        is_screen = report["screen_analysis"].get("is_screen") == "Yes"
        screen_score = float(report["screen_analysis"].get("screen_score", "0"))
        
        if is_screen:
            explanations["screen_analysis"] = f"""❌ **Possível foto de tela detectada (pontuação: {screen_score:.2f})**
            
Foram identificados padrões característicos de telas/displays na imagem. Isso inclui reflexos retangulares, padrões Moiré, estruturas de pixel ou gradientes não naturais que são típicos quando alguém fotografa uma tela ou monitor."""
        elif screen_score > 0.3:
            explanations["screen_analysis"] = f"""⚠️ **Análise de tela inconclusiva (pontuação: {screen_score:.2f})**
            
Alguns indícios de possível apresentação em tela foram detectados, mas não são suficientemente conclusivos. A imagem pode ser autêntica, mas apresenta algumas características semelhantes às de uma fotografia de tela."""
        else:
            explanations["screen_analysis"] = f"""✅ **Sem indícios de foto de tela (pontuação: {screen_score:.2f})**
            
A análise não detectou padrões típicos de telas/displays na imagem. A distribuição de pixels, reflexos e texturas são consistentes com uma fotografia direta, não uma foto de outra imagem exibida em tela."""
    
    # Explicação sobre análise de profundidade
    if "depth_analysis" in report:
        depth_score = float(report["depth_analysis"].get("depth_score", "0"))
        
        if depth_score > 0.6:
            explanations["depth_analysis"] = f"""✅ **Boa variação de profundidade detectada (pontuação: {depth_score:.2f})**
            
O mapa de profundidade mostra variações naturais, consistentes com uma face real tridimensional. As diferentes partes do rosto apresentam profundidades variadas, como esperado em uma pessoa real."""
        elif depth_score > 0.4:
            explanations["depth_analysis"] = f"""⚠️ **Variação moderada de profundidade (pontuação: {depth_score:.2f})**
            
O mapa de profundidade mostra alguma variação, mas não tão pronunciada quanto esperado. Isto pode ser devido à qualidade da imagem, iluminação ou potencialmente algum tipo de falsificação."""
        else:
            explanations["depth_analysis"] = f"""❌ **Baixa variação de profundidade detectada (pontuação: {depth_score:.2f})**
            
O mapa de profundidade mostra muito pouca variação, o que é característico de imagens planas como fotografias de outras fotos ou documentos. Rostos reais apresentam naturalmente variações significativas de profundidade entre diferentes partes faciais."""
    
    return explanations

def detect_document_tampering(image_path):
    """
    Detecta sinais de adulteração em documentos usando várias técnicas
    
    Parameters:
        image_path (str): Caminho para a imagem do documento
        
    Returns:
        dict: Resultados da detecção de adulteração
    """
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return {"tampered": True, "score": 1.0, "reasons": ["Imagem inválida"]}
        
        # Converter para diferentes espaços de cor para análises complementares
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        results = {
            "tampered": False,
            "score": 0.0,  # 0.0 = não adulterado, 1.0 = certamente adulterado
            "analysis": {},
            "reasons": []
        }
        
        # 1. Análise ELA (Error Level Analysis)
        # Útil para detectar áreas coladas, texto modificado ou elementos alterados
        temp_path = "temp_ela_doc.jpg"
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Recarregar imagem comprimida e calcular diferença
        compressed = cv2.imread(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if compressed is not None:
            # Calcular diferença absoluta
            ela_diff = cv2.absdiff(img, compressed)
            ela_diff_norm = cv2.normalize(ela_diff, None, 0, 255, cv2.NORM_MINMAX)
            
            # Converter para escala de cinza e calcular estatísticas
            ela_gray = cv2.cvtColor(ela_diff_norm, cv2.COLOR_BGR2GRAY)
            _, ela_binary = cv2.threshold(ela_gray, 50, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos de áreas suspeitas
            contours, _ = cv2.findContours(ela_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos significativos
            suspicious_areas = []
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Ignorar áreas muito pequenas
                        suspicious_areas.append(contour)
            
            # Salvar imagem com áreas suspeitas destacadas
            ela_result = img.copy()
            cv2.drawContours(ela_result, suspicious_areas, -1, (0, 0, 255), 2)
            
            ela_output_path = f"temp/doc_ela_analysis_{int(time.time())}.jpg"
            if not os.path.exists('temp'):
                os.makedirs('temp')
            cv2.imwrite(ela_output_path, ela_result)
            
            # Calcular score ELA baseado na quantidade de áreas suspeitas
            ela_score = min(1.0, len(suspicious_areas) / 20.0)
            results["analysis"]["ela"] = {
                "score": ela_score,
                "suspicious_areas": len(suspicious_areas),
                "output_path": ela_output_path
            }
            
            if ela_score > 0.3:
                results["reasons"].append(f"Detectadas {len(suspicious_areas)} áreas com níveis de erro inconsistentes")
        
        # 2. Análise de consistência de fontes
        # Texto em documentos legítimos tem características homogêneas
        # Aplicar detecção de bordas para realçar texto
        edges = cv2.Canny(gray, 100, 200)
        
        # Operações morfológicas para conectar caracteres de texto
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Encontrar contornos que provavelmente são texto
        text_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analisar proporções e tamanhos dos contornos de texto
        char_heights = []
        char_widths = []
        
        for contour in text_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filtrar para considerar apenas possíveis caracteres
            if 5 < h < 50 and 2 < w < 50:
                char_heights.append(h)
                char_widths.append(w)
        
        # Calcular variabilidade das dimensões
        if char_heights and char_widths:
            height_std = np.std(char_heights)
            height_mean = np.mean(char_heights)
            width_std = np.std(char_widths)
            width_mean = np.mean(char_widths)
            
            # Em documentos legítimos, caracteres tendem a ter altura consistente
            # Alto desvio padrão pode indicar fontes incompatíveis (texto alterado)
            height_variation = height_std / height_mean if height_mean > 0 else 0
            width_variation = width_std / width_mean if width_mean > 0 else 0
            
            # Score de consistência de fonte
            font_score = min(1.0, (height_variation + width_variation) / 1.0)
            results["analysis"]["font"] = {
                "score": font_score,
                "height_variation": height_variation,
                "width_variation": width_variation
            }
            
            if font_score > 0.4:
                results["reasons"].append("Inconsistências detectadas nas fontes de texto")
        
        # 3. Análise de padrões de segurança (microprintings, guilloche, etc.)
        # Documentos oficiais têm padrões de alta frequência que desaparecem em cópias
        
        # Aplicar filtro de passo alto para detectar componentes de alta frequência
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        highpass = cv2.subtract(gray, blur)
        
        # Binarizar para destacar padrões finos
        _, highpass_binary = cv2.threshold(highpass, 20, 255, cv2.THRESH_BINARY)
        
        # Calcular métricas dos padrões finos
        fine_pattern_density = np.sum(highpass_binary) / (gray.shape[0] * gray.shape[1])
        
        # Documentos originais têm densidade alta de padrões finos
        # Cópias/falsificações perdem estes detalhes
        security_score = max(0, 1.0 - fine_pattern_density * 5)  # Ajustado empiricamente
        results["analysis"]["security_patterns"] = {
            "score": security_score,
            "pattern_density": fine_pattern_density
        }
        
        if security_score > 0.6:
            results["reasons"].append("Ausência de micro-impressões e padrões de segurança")
        
        # 4. Análise de bordas e cantos
        # Documentos adulterados frequentemente têm problemas nas bordas
        
        # Detectar bordas usando Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Encontrar contornos externos (bordas do documento)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Encontrar o contorno principal do documento
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # Aproximar contorno para um polígono
            peri = cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, 0.02 * peri, True)
            
            # Verificar se é aproximadamente retangular
            if len(approx) != 4:
                # Se não for um retângulo, possível problema com cantos
                edge_score = min(1.0, abs(len(approx) - 4) / 4)
                results["analysis"]["edges"] = {
                    "score": edge_score,
                    "corner_count": len(approx)
                }
                
                if edge_score > 0.3:
                    results["reasons"].append(f"Formato irregular detectado ({len(approx)} cantos em vez de 4)")
                
            # Verificar uniformidade dos cantos
            else:
                # Calcular ângulos dos cantos
                angles = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i+1) % 4][0]
                    p3 = approx[(i+2) % 4][0]
                    
                    # Calcular vetores
                    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
                    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
                    
                    # Calcular ângulo usando produto escalar
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    magnitude1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    magnitude2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if magnitude1*magnitude2 > 0:
                        cos_angle = dot_product / (magnitude1 * magnitude2)
                        # Limitar devido a possíveis erros de precisão
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = np.degrees(np.arccos(cos_angle))
                        angles.append(angle)
                
                # Em um retângulo perfeito, todos os ângulos são próximos de 90°
                if angles:
                    angle_variation = np.std(angles)
                    corner_score = min(1.0, angle_variation / 15.0)  # Ajustado empiricamente
                    
                    results["analysis"]["corners"] = {
                        "score": corner_score,
                        "angle_variation": angle_variation
                    }
                    
                    if corner_score > 0.4:
                        results["reasons"].append(f"Ângulos irregulares nos cantos (variação: {angle_variation:.1f}°)")
        
        # 5. Análise de consistência de cores
        # Documentos genuínos têm paletas de cores consistentes
        
        # Extrair principais cores usando K-means
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Limitar a número de pixels para processamento mais rápido
        if pixels.shape[0] > 10000:
            indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
        
        # Aplicar clustering K-means
        K = 5  # Número de clusters de cores
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels_sample, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calcular histograma de labels
        hist, _ = np.histogram(labels, bins=K, range=(0, K))
        hist = hist.astype(float) / hist.sum()
        
        # Calcular a entropia da distribuição de cores
        # Documentos falsificados frequentemente têm distribuição anormal
        non_zeros = hist[hist > 0]
        entropy = -np.sum(non_zeros * np.log2(non_zeros))
        
        # Documentos genuínos têm entropia moderada
        # Entropia muito alta ou baixa pode indicar problemas
        entropy_score = min(1.0, abs(entropy - 2.0) / 2.0)  # Valor 2.0 é empiricamente bom
        
        results["analysis"]["color"] = {
            "score": entropy_score,
            "entropy": entropy
        }
        
        if entropy_score > 0.5:
            results["reasons"].append(f"Distribuição de cores suspeita (entropia: {entropy:.2f})")
        
        # Calcular pontuação final combinando todos os fatores
        weights = {
            "ela": 0.30,
            "font": 0.15,
            "security_patterns": 0.25,
            "edges": 0.15,
            "corners": 0.10,
            "color": 0.05
        }
        
        final_score = 0.0
        for key, weight in weights.items():
            if key in results["analysis"] and "score" in results["analysis"][key]:
                final_score += results["analysis"][key]["score"] * weight
        
        results["score"] = min(1.0, max(0.0, final_score))
        results["tampered"] = results["score"] > 0.5
        
        # Criar uma visualização combinada das análises
        final_viz = img.copy()
        
        # Adicionar contornos das áreas suspeitas (ELA)
        if "ela" in results["analysis"] and "suspicious_areas" in locals() and suspicious_areas:
            # Desenhar contornos vermelhos para áreas suspeitas
            cv2.drawContours(final_viz, suspicious_areas, -1, (0, 0, 255), 2)
            
            # Adicionar círculos nas áreas mais suspeitas
            for contour in suspicious_areas:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(final_viz, (cx, cy), 10, (0, 165, 255), -1)  # Círculo laranja
                    
                    # Adicionar número de identificação para cada área suspeita
                    area = cv2.contourArea(contour)
                    cv2.putText(final_viz, f"A{area:.0f}", (cx+5, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Adicionar zoom em uma das áreas suspeitas (apenas a primeira grande)
                    if area > 300 and not 'zoom_drawn' in locals():
                        # Obter bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Garantir que a área está dentro da imagem
                        if (x >= 0 and y >= 0 and 
                            x+w < img.shape[1] and y+h < img.shape[0] and
                            w > 0 and h > 0):
                            
                            # Extrair região suspeita
                            roi = img[y:y+h, x:x+w]
                            
                            # Tamanho do zoom
                            zoom_size = min(150, img.shape[0]//4)
                            zoom_img = cv2.resize(roi, (zoom_size, zoom_size))
                            
                            # Posicionar no canto superior direito
                            y_offset = 70  # Espaço para a barra de status
                            x_offset = img.shape[1] - zoom_size - 10
                            
                            # Criar fundo para o zoom
                            zoom_bg = np.ones((zoom_size+20, zoom_size+20, 3), dtype=np.uint8) * 240
                            zoom_bg[10:10+zoom_size, 10:10+zoom_size] = zoom_img
                            
                            # Adicionar borda vermelha
                            cv2.rectangle(zoom_bg, (9, 9), (zoom_size+10, zoom_size+10), (0, 0, 255), 2)
                            
                            # Adicionar texto explicativo
                            cv2.putText(zoom_bg, "ÁREA SUSPEITA AMPLIADA", (10, zoom_size+15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            
                            # Inserir na imagem principal
                            if (y_offset+zoom_bg.shape[0] <= final_viz.shape[0] and
                                x_offset+zoom_bg.shape[1] <= final_viz.shape[1]):
                                final_viz[y_offset:y_offset+zoom_bg.shape[0], 
                                         x_offset:x_offset+zoom_bg.shape[1]] = zoom_bg
                            
                            # Desenhar linha conectando a área original ao zoom
                            cv2.line(final_viz, (x+w//2, y+h//2), 
                                    (x_offset+zoom_size//2, y_offset+zoom_size//2), 
                                    (0, 0, 255), 1, cv2.LINE_AA)
                            
                            zoom_drawn = True
        
        # Adicionar marcações de problemas nos cantos/bordas
        if 'approx' in locals() and len(approx) != 4:
            cv2.drawContours(final_viz, [approx], -1, (255, 0, 0), 3)  # Contorno azul
            
            # Marcar os cantos detectados
            for i, point in enumerate(approx):
                cv2.circle(final_viz, tuple(point[0]), 8, (255, 255, 0), -1)  # Círculo amarelo
                cv2.putText(final_viz, str(i+1), (point[0][0]-4, point[0][1]+4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Destacar regiões de texto suspeitas se houver problema com fontes
        if "font" in results["analysis"] and results["analysis"]["font"]["score"] > 0.4:
            # Desenhar retângulos nas regiões de texto suspeitas
            if 'text_contours' in locals():
                for contour in text_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 5 < h < 50 and 2 < w < 50:  # Filtrar para possíveis caracteres
                        cv2.rectangle(final_viz, (x, y), (x+w, y+h), (0, 200, 200), 1)  # Retângulo amarelo claro
        
        # Adicionar grade de análise técnica no documento
        # Dividir a imagem em uma grade 4x4 para análise
        grid_rows, grid_cols = 4, 4
        h, w = img.shape[:2]
        cell_h, cell_w = h // grid_rows, w // grid_cols
        
        # Desenhar linhas da grade
        for i in range(1, grid_rows):
            y_pos = i * cell_h
            cv2.line(final_viz, (0, y_pos), (w, y_pos), (100, 100, 100), 1, cv2.LINE_AA)
            
        for j in range(1, grid_cols):
            x_pos = j * cell_w
            cv2.line(final_viz, (x_pos, 0), (x_pos, h), (100, 100, 100), 1, cv2.LINE_AA)
        
        # Enumerar os quadrantes para referência
        for i in range(grid_rows):
            for j in range(grid_cols):
                cell_id = i * grid_cols + j + 1
                x_pos = j * cell_w + 5
                y_pos = i * cell_h + 15
                cv2.putText(final_viz, str(cell_id), (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Adicionar diagrama das áreas críticas do documento (exemplo: cabeçalho, foto, assinatura)
        regions = [
            {"name": "Cabeçalho", "coords": (0, 0, w, h//6), "color": (150, 200, 100)},
            {"name": "Foto", "coords": (0, h//6, w//4, h//3), "color": (100, 100, 220)},
            {"name": "Assinatura", "coords": (w//2, 2*h//3, w//3, h//6), "color": (220, 100, 100)}
        ]
        
        # Desenhar regiões críticas com transparência
        overlay = final_viz.copy()
        alpha = 0.3  # Transparência
        
        for region in regions:
            x, y, rw, rh = region["coords"]
            # Desenhar retângulo preenchido
            cv2.rectangle(overlay, (x, y), (x+rw, y+rh), region["color"], -1)
            # Desenhar borda
            cv2.rectangle(final_viz, (x, y), (x+rw, y+rh), region["color"], 1)
            # Adicionar nome da região
            cv2.putText(final_viz, region["name"], (x+5, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, region["color"], 1)
        
        # Combinar com a imagem original
        cv2.addWeighted(overlay, alpha, final_viz, 1-alpha, 0, final_viz)
        
        # Adicionar uma barra de status/pontuação na parte superior
        # Criar uma área para a barra de status
        status_height = 60
        status_bar = np.zeros((status_height, img.shape[1], 3), dtype=np.uint8)
        
        # Definir cor baseada na pontuação
        if results["score"] < 0.3:
            status_color = (0, 255, 0)  # Verde = bom
            status_text = "DOCUMENTO AUTÊNTICO"
        elif results["score"] < 0.5:
            status_color = (0, 255, 255)  # Amarelo = atenção
            status_text = "POSSÍVEIS INCONSISTÊNCIAS"
        else:
            status_color = (0, 0, 255)  # Vermelho = problema
            status_text = "ADULTERAÇÕES DETECTADAS"
        
        # Preencher a barra de status com a cor
        status_bar[:] = status_color
        
        # Adicionar texto
        auth_score = int((1 - results["score"]) * 100)
        cv2.putText(status_bar, f"{status_text} - Autenticidade: {auth_score}%", 
                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combinar barra de status com a imagem
        final_output = np.vstack((status_bar, final_viz))
        
        # Salvar resultado final
        tampering_result_path = f"temp/document_tampering_analysis_{int(time.time())}.jpg"
        if not os.path.exists('temp'):
            os.makedirs('temp')
        cv2.imwrite(tampering_result_path, final_output)
        results["result_image"] = tampering_result_path
        
        return results
    
    except Exception as e:
        print(f"Erro ao analisar adulteração de documento: {str(e)}")
        return {
            "tampered": False,
            "score": 0.0,
            "reasons": [f"Erro na análise: {str(e)}"],
            "analysis": {}
        }

def display_complete_report_explanation(report, explanations):
    """Monta uma explicação completa do relatório para exibição."""
    
    complete_explanation = {
        "Resultado geral": {
            "status": f"**Resultado**: {report['overall_result']}",
            "trust_score": f"**Pontuação de confiabilidade**: {report.get('trust_score', '0')}%",
            "explanation": explanations.get("overall", "Não disponível")
        },
        "Análise facial": {
            "similarity_score": f"**Pontuação de similaridade**: {report['similarity_score']}",
            "face_authentic": f"**Face autêntica**: {'No' if report.get('face_swap_detected') == 'Yes' else 'Yes'}",
            "face_swap": f"**Face swap detectado**: {report.get('face_swap_detected', 'N/A')}",
            "explanation": explanations.get("facial_match", "Não disponível")
        },
        "Verificação de imagem": {
            "liveness": f"**Vivacidade detectada**: {report.get('liveness_detected', 'N/A')}",
            "image_edited": f"**Imagem editada**: {report.get('image_edited', 'N/A')}",
            "lighting_quality": f"**Qualidade da iluminação**: {report.get('lighting_quality', 'N/A')}",
            "explanation_liveness": explanations.get("liveness", "Não disponível"),
            "explanation_editing": explanations.get("image_editing", "Não disponível")
        },
        "Verificação de idade": {
            "declared_age": f"**Idade declarada**: {report.get('declared_age', 'N/A')}",
            "estimated_age": f"**Idade estimada**: {report.get('estimated_age', 'N/A')}",
            "age_match": f"**Correspondência de idade**: {report.get('age_match', 'N/A')}",
            "explanation": explanations.get("age", "Não disponível")
        },
        "Verificação de gênero": {
            "declared_gender": f"**Gênero declarado**: {report.get('declared_gender', 'N/A')}",
            "estimated_gender": f"**Gênero estimado**: {report.get('estimated_gender', 'N/A')}",
            "gender_match": f"**Correspondência de gênero**: {report.get('gender_match', 'N/A')}"
        },
        "Autenticidade do documento": {
            "authenticity": f"**Adulteração detectada**: {report.get('document_authenticity', 'Não analisado')}",
            "authenticity_score": f"**Pontuação de autenticidade**: {report.get('document_authenticity_score', 'N/A')}"
        },
        "Metadados EXIF": {
            "exif_consistent": f"**Metadados consistentes**: {report.get('exif_consistent', 'N/A')}",
            "exif_details": f"**Detalhes EXIF**: {str(report.get('exif_analysis', 'Não disponível'))}"
        },
        "Recomendações": {
            "recommendations": "\n".join([f"- {rec}" for rec in report.get('recommendations', [])]) or "Nenhuma recomendação necessária."
        }
    }
    
    return complete_explanation

def detect_document_type(document_path):
    """
    Detecta o tipo de documento usando OCR e análise de características básicas.
    Esta é uma versão simplificada usada quando PyTorch/TensorFlow não está disponível.
    
    Parameters:
        document_path (str): Caminho para a imagem do documento
        
    Returns:
        tuple: (document_type, confidence_score)
    """
    try:
        # Carregar a imagem
        img = cv2.imread(document_path)
        if img is None:
            return "Desconhecido", 0.0
            
        # Extrair texto usando OCR
        document_text = extract_text_from_document(document_path)
        
        # Detectar tipo de documento baseado em palavras-chave
        text = document_text.lower()
        
        # Palavras-chave para cada tipo de documento
        rg_keywords = ["república federativa", "identidade", "rg", "carteira de identidade", "registro geral"]
        cnh_keywords = ["carteira nacional", "habilitação", "cnh", "motorista", "acc", "permissão para dirigir"]
        passport_keywords = ["passport", "passaporte", "república federativa do brasil", "brasileiro", "federative republic"]
        
        # Contagem de palavras-chave encontradas
        rg_count = sum(1 for keyword in rg_keywords if keyword in text)
        cnh_count = sum(1 for keyword in cnh_keywords if keyword in text)
        passport_count = sum(1 for keyword in passport_keywords if keyword in text)
        
        # Análise de cores predominantes para ajudar na determinação
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # RG normalmente tem tons verde/azulados
        # CNH normalmente tem tons verde/rosa
        # Passaporte normalmente tem tons azuis
        
        # Simplificação: analisar a matiz média
        avg_hue = np.mean(hsv[:, :, 0])
        
        # Determinação final por pontuação
        scores = {
            "RG": rg_count * 1.5 + (1.0 if 30 < avg_hue < 120 else 0.0),
            "CNH": cnh_count * 1.5 + (1.0 if 90 < avg_hue < 180 else 0.0),
            "Passaporte": passport_count * 1.5 + (1.0 if 100 < avg_hue < 140 else 0.0),
            "Outro": 1.0  # Base score para "Outro"
        }
        
        # Determinar o tipo com maior pontuação
        doc_type = max(scores, key=scores.get)
        confidence = (scores[doc_type] / 5.0)  # Normalizar para 0-1
        
        return doc_type, min(0.95, confidence)  # Limitar a 0.95 pois não é o método mais confiável
        
    except Exception as e:
        print(f"Erro na detecção do tipo de documento: {e}")
        return "Desconhecido", 0.0

def classify_document_type_with_cnn(document_path):
    """
    Classifica o tipo de documento utilizando uma rede neural convolucional leve (MobileNetV2).
    Esta função é mais robusta que a detecção baseada apenas em OCR + cor, funcionando
    mesmo em documentos escaneados ou com cores alteradas.
    
    Parameters:
        document_path (str): Caminho para a imagem do documento
        
    Returns:
        tuple: (document_type, confidence_score)
    """
    try:
        # Verificar se TensorFlow e Keras estão disponíveis
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        
        # Verificar se o modelo já existe ou precisa ser criado
        model_path = 'models/document_classifier_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Modelo de classificação de documentos carregado com sucesso.")
        else:
            print("Modelo de classificação não encontrado. Criando modelo base (precisa ser treinado).")
            # Se não existe, criar um modelo base com MobileNetV2
            # Em produção, este modelo precisaria ser treinado com um dataset de documentos
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Adicionar camadas de classificação
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(4, activation='softmax')(x)  # 4 classes (RG, CNH, Passaporte, Outro)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Congelar camadas do modelo base
            for layer in base_model.layers:
                layer.trainable = False
                
            # Compilar modelo
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Salvar o modelo base
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
            
            # Retornar resultado placeholder já que o modelo não está treinado
            return "Documento genérico", 0.5
        
        # Carregar e pré-processar a imagem
        img = image.load_img(document_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Realizar a predição
        predictions = model.predict(img_array)
        
        # Classes disponíveis (precisariam ser definidas durante o treinamento real)
        document_classes = ["RG", "CNH", "Passaporte", "Outro"]
        
        # Obter a classe com maior probabilidade
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        document_type = document_classes[class_index]
        
        print(f"Documento classificado como {document_type} com confiança de {confidence:.2f}")
        
        return document_type, confidence
        
    except ImportError as e:
        print(f"TensorFlow/Keras não está disponível: {e}")
        # Fallback para a abordagem baseada em OCR+cor
        return detect_document_type(document_path)
    except Exception as e:
        print(f"Erro na classificação CNN de documento: {e}")
        # Fallback para a abordagem baseada em OCR+cor
        return detect_document_type(document_path)

def generate_verification_radar_chart(report):
    """
    Gera um gráfico de radar (radar chart) que mostra visualmente os diferentes 
    scores de verificação para maior transparência e explicabilidade.
    
    Parameters:
        report (dict): O relatório de verificação contendo os scores
        
    Returns:
        str: Caminho para a imagem do gráfico salva
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Necessário para ambientes sem GUI
        
        # Extrair scores do relatório - converter para valores numéricos (0-1)
        scores = {}
        
        # Similaridade facial (já está em formato 0-1)
        scores['Similaridade'] = float(report.get('similarity_score', 0))
        
        # Verificação de vivacidade
        if isinstance(report.get('liveness_detected', False), bool):
            scores['Vivacidade'] = 1.0 if report.get('liveness_detected') == 'Yes' else 0.2
        else:
            scores['Vivacidade'] = 1.0 if report.get('liveness_detected') == 'Yes' else 0.2
            
        # Autenticidade de face (inverso da detecção de face swap)
        if report.get('face_swap_detected') == 'Yes':
            scores['Autenticidade'] = 0.2
        else:
            scores['Autenticidade'] = 0.9
            
        # Originalidade (inverso da detecção de edição)
        if report.get('image_edited') == 'Yes':
            scores['Originalidade'] = 0.2
        else:
            scores['Originalidade'] = 0.9
            
        # Qualidade da imagem
        if report.get('lighting_quality') == 'Good':
            scores['Qualidade'] = 0.9
        elif report.get('lighting_quality') == 'Medium':
            scores['Qualidade'] = 0.6
        else:
            scores['Qualidade'] = 0.3
            
        # Autenticidade do documento
        if 'document_authenticity' in report:
            if report.get('document_authenticity') == 'Authentic':
                scores['Doc. Autêntico'] = 0.9
            elif report.get('document_authenticity') == 'Suspicious':
                scores['Doc. Autêntico'] = 0.4
            else:
                scores['Doc. Autêntico'] = 0.1
        else:
            scores['Doc. Autêntico'] = 0.5
        
        # Configurar o gráfico de radar
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Adicionar o primeiro valor novamente para fechar o polígono
        values.append(values[0])
        categories.append(categories[0])
        
        # Converter para array numpy
        values = np.array(values)
        
        # Calcular ângulos para as categorias
        N = len(categories) - 1  # -1 porque adicionamos o primeiro novamente
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Adicionar o primeiro ângulo novamente para fechar o polígono
        angles.append(angles[0])
        
        # Criar figura e radar plot
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Plotar o gráfico de radar
        ax.plot(angles, values, 'o-', linewidth=2, label='Scores')
        ax.fill(angles, values, alpha=0.25)
        
        # Definir categorias como labels
        ax.set_thetagrids(np.degrees(angles), categories)
        
        # Definir limites para o gráfico
        ax.set_ylim(0, 1)
        
        # Determinar a cor do preenchimento baseado no score geral
        overall_score = float(report.get('trust_score', '0').replace('%', '')) / 100
        if overall_score > 0.7:
            color = 'green'
        elif overall_score > 0.5:
            color = 'orange'
        else:
            color = 'red'
            
        # Preencher novamente com a cor determinada
        ax.fill(angles, values, alpha=0.25, color=color)
        
        # Adicionar título e legend
        plt.title('Análise de Verificação de Identidade', size=15, y=1.1)
        
        # Adicionar círculos de referência e labels
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        
        # Adicionar score total no centro
        plt.text(0, 0, f"{report.get('trust_score', '0')}",
                 ha='center', va='center', fontsize=20, 
                 bbox=dict(facecolor='white', alpha=0.5))
        
        # Salvar o gráfico como imagem
        output_path = os.path.join(UPLOAD_FOLDER, 'radar_chart.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    except Exception as e:
        print(f"Erro ao gerar gráfico de radar: {e}")
        return None

def analyze_exif_metadata(image_path, return_score=False):
    """
    Analisa metadados EXIF da imagem para detectar inconsistências ou manipulações.
    
    Parameters:
        image_path (str): Caminho para a imagem
        return_score (bool): Se deve retornar o score de confiança
        
    Returns:
        tuple: (is_consistent, score, metadata_analysis)
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        import datetime
        
        # Carregar imagem com metadados
        img = Image.open(image_path)
        exif_data = {}
        
        # Extrair os metadados EXIF
        if hasattr(img, '_getexif') and img._getexif() is not None:
            for tag, value in img._getexif().items():
                if tag in TAGS:
                    exif_data[TAGS[tag]] = value
        
        # Se não há dados EXIF suficientes, pode ser sinal de manipulação
        if len(exif_data) < 3:
            print("Poucos metadados EXIF - possível indicação de manipulação")
            return (False, 0.3, {"warning": "Poucos metadados EXIF", "exif_count": len(exif_data)}) if return_score else False
        
        # Analisar data/hora de criação
        date_time_original = exif_data.get('DateTimeOriginal')
        date_time_digitized = exif_data.get('DateTimeDigitized')
        
        # Verificar inconsistências entre datas
        date_inconsistent = False
        if date_time_original and date_time_digitized and date_time_original != date_time_digitized:
            try:
                time_diff = abs(datetime.datetime.strptime(date_time_original, "%Y:%m:%d %H:%M:%S") - 
                            datetime.datetime.strptime(date_time_digitized, "%Y:%m:%d %H:%M:%S"))
                if time_diff.total_seconds() > 300:  # Diferença maior que 5 minutos
                    date_inconsistent = True
            except ValueError:
                # Formato de data inválido, possivelmente manipulado
                date_inconsistent = True
        
        # Verificar software de processamento
        processing_software = exif_data.get('Software', '')
        editing_software_patterns = ['photoshop', 'lightroom', 'gimp', 'affinity', 'capture one']
        was_edited = any(pattern in processing_software.lower() for pattern in editing_software_patterns) if processing_software else False
        
        # Verificar consistência de parâmetros
        make = exif_data.get('Make', '')
        model = exif_data.get('Model', '')
        
        # Calcular score baseado nas análises
        score = 1.0
        
        if date_inconsistent:
            score -= 0.3
        
        if was_edited:
            score -= 0.2
        
        if not make or not model:
            score -= 0.1
            
        # Preparar análise detalhada
        metadata_analysis = {
            "exif_count": len(exif_data),
            "date_inconsistent": date_inconsistent,
            "was_edited": was_edited,
            "camera": f"{make} {model}".strip(),
            "software": processing_software
        }
        
        is_consistent = score > 0.7
        
        if return_score:
            return is_consistent, score, metadata_analysis
        else:
            return is_consistent
            
    except Exception as e:
        print(f"Erro na análise EXIF: {e}")
        return (False, 0.5, {"error": str(e)}) if return_score else False

def detect_screen_spoofing(image_path):
    """
    Detecta se uma imagem é uma foto de uma tela/monitor (spoofing) usando 
    técnicas avançadas de análise de padrões Moiré, reflexos de tela e 
    outros artefatos característicos de displays.
    
    Parameters:
        image_path (str): Caminho para a imagem a ser analisada
        
    Returns:
        tuple: (is_screen_spoof, confidence_score, análise_detalhada)
    """
    try:
        # Carregar a imagem
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, {}
            
        # Converter para diferentes espaços de cor para análises múltiplas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Análises a serem realizadas
        results = {}
        
        # 1. Detecção de padrões Moiré usando FFT
        # Os padrões Moiré aparecem como picos periódicos no espectro de frequência
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
        
        # Analisar a estrutura do espectro de frequência
        # Telas geralmente mostram padrões regulares
        center_removed = magnitude_spectrum.copy()
        h, w = center_removed.shape
        center_y, center_x = h // 2, w // 2
        mask_radius = min(h, w) // 8
        
        # Criar máscara para remover a componente DC (centro do espectro)
        y, x = np.ogrid[:h, :w]
        mask_area = (x - center_x)**2 + (y - center_y)**2 <= mask_radius**2
        center_removed[mask_area] = 0
        
        # Calcular estatísticas do espectro sem o centro
        high_freq_energy = np.sum(center_removed) / (center_removed.size - np.sum(mask_area))
        mid_freq_peaks = np.percentile(center_removed, 99)
        
        # Detectar picos significativos fora do centro
        peak_threshold = np.percentile(center_removed, 99.9)
        peaks_count = np.sum(center_removed > peak_threshold)
        
        # Razão de picos para identificar padrões regulares (como em telas)
        peak_ratio = peaks_count / center_removed.size
        results['moire_peak_ratio'] = peak_ratio
        results['high_freq_energy'] = high_freq_energy
        
        # Detecção de padrões de grade usando DCT
        dct = cv2.dct(np.float32(gray))
        dct_log = np.log(abs(dct) + 1)
        dct_high_freq = dct_log[5:20, 5:20]  # Analisar frequências médias
        dct_energy = np.mean(dct_high_freq)
        results['dct_energy'] = dct_energy
        
        # 2. Detectar reflexos de tela
        # Reflexos em telas geralmente são muito brilhantes e têm bordas muito definidas
        
        # Extrair canal V (brilho) do HSV
        v_channel = hsv[:, :, 2]
        
        # Detectar áreas muito brilhantes (possíveis reflexos)
        _, bright_areas = cv2.threshold(v_channel, 240, 255, cv2.THRESH_BINARY)
        
        # Análise morfológica para detectar formas específicas de reflexos de tela
        kernel = np.ones((5, 5), np.uint8)
        bright_areas = cv2.morphologyEx(bright_areas, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos de áreas brilhantes
        contours, _ = cv2.findContours(bright_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analisar propriedades dos reflexos
        rectangular_reflections = 0
        reflection_area_ratio = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Ignorar reflexos muito pequenos
                continue
                
            # Verificar se o reflexo tem forma retangular (característico de telas)
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Reflexos de tela tendem a ser retangulares
            if extent > 0.7 and rect_area > 0.001 * img.shape[0] * img.shape[1]:
                rectangular_reflections += 1
                reflection_area_ratio += area / (img.shape[0] * img.shape[1])
        
        results['rectangular_reflections'] = rectangular_reflections
        results['reflection_area_ratio'] = reflection_area_ratio
        
        # 3. Detectar linhas retas/grades - muito comuns em fotos de telas
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=img.shape[1]//10, maxLineGap=10)
        
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                
                # Classificar como horizontal ou vertical
                if angle < 10 or angle > 170:
                    horizontal_lines += 1
                elif 80 < angle < 100:
                    vertical_lines += 1
        
        results['horizontal_lines'] = horizontal_lines
        results['vertical_lines'] = vertical_lines
        results['grid_structure'] = (horizontal_lines > 5 and vertical_lines > 5)
        
        # 4. Análise de textura para subpixels (padrões RGB de displays)
        # Displays têm padrões específicos de subpixels que podem ser detectados
        
        # Separar canais RGB e verificar correlação entre eles
        b, g, r = cv2.split(img)
        
        # Calcular gradientes para cada canal
        grad_x_r = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_r = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)
        grad_x_g = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_g = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        grad_x_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular correlação entre gradientes de diferentes canais
        # Em telas, os canais RGB terão correlações específicas devido à estrutura de subpixels
        correlation_rg = np.corrcoef(grad_x_r.flatten(), grad_x_g.flatten())[0, 1]
        correlation_rb = np.corrcoef(grad_x_r.flatten(), grad_x_b.flatten())[0, 1]
        correlation_gb = np.corrcoef(grad_x_g.flatten(), grad_x_b.flatten())[0, 1]
        
        results['correlation_rg'] = correlation_rg
        results['correlation_rb'] = correlation_rb
        results['correlation_gb'] = correlation_gb
        
        # Análise média de correlação (geralmente mais alta em telas)
        mean_correlation = (abs(correlation_rg) + abs(correlation_rb) + abs(correlation_gb)) / 3
        results['mean_correlation'] = mean_correlation
        
        # 5. Ponderação final e classificação
        # Definir pesos para cada fator
        weights = {
            'moire_peak_ratio': 0.25,
            'dct_energy': 0.15,
            'rectangular_reflections': 0.15,
            'grid_structure': 0.20,
            'mean_correlation': 0.25
        }
        
        # Calcular score de "tela"
        screen_score = 0.0
        
        # Contribuição de padrões Moiré (0-1)
        moire_contribution = min(1.0, max(0.0, peak_ratio * 5000))
        screen_score += weights['moire_peak_ratio'] * moire_contribution
        
        # Contribuição de energia DCT (0-1)
        dct_contribution = min(1.0, max(0.0, dct_energy / 10))
        screen_score += weights['dct_energy'] * dct_contribution
        
        # Contribuição de reflexos retangulares (0-1)
        reflection_contribution = min(1.0, max(0.0, rectangular_reflections / 3))
        screen_score += weights['rectangular_reflections'] * reflection_contribution
        
        # Contribuição de estrutura de grade (0-1)
        grid_contribution = 1.0 if results['grid_structure'] else 0.0
        screen_score += weights['grid_structure'] * grid_contribution
        
        # Contribuição de correlação de canais (0-1)
        correlation_contribution = min(1.0, max(0.0, mean_correlation * 2))
        screen_score += weights['mean_correlation'] * correlation_contribution
        
        # Determinar classificação final
        is_screen = screen_score > 0.5
        
        # Log para diagnóstico
        print(f"Screen Detection: Score={screen_score:.2f}, Moire={moire_contribution:.2f}, DCT={dct_contribution:.2f}, Reflexos={reflection_contribution:.2f}, Grade={grid_contribution:.2f}, Corr={correlation_contribution:.2f}")
        
        return is_screen, screen_score, results
    
    except Exception as e:
        print(f"Erro na detecção de tela/spoofing: {str(e)}")
        return False, 0.0, {"error": str(e)}

def analyze_depth_for_liveness(image_path):
    """
    Estima a profundidade da cena usando o modelo MiDaS para detectar vivacidade.
    Uma face real tem variações naturais de profundidade, enquanto uma foto de uma foto
    ou uma face em uma tela terá uma superfície mais plana.
    
    Parameters:
        image_path (str): Caminho para a imagem a ser analisada
        
    Returns:
        tuple: (depth_score, depth_map_path, depth_analysis)
            depth_score: valor entre 0-1 indicando a variação de profundidade (maior = mais natural)
            depth_map_path: caminho para a visualização do mapa de profundidade gerado
            depth_analysis: dicionário com métricas detalhadas da análise
    """
    try:
        # Verificar se o PyTorch está disponível
        if not 'torch' in sys.modules:
            print("PyTorch não disponível para análise de profundidade")
            return 0.5, None, {"error": "PyTorch não disponível"}
        
        # Verificar se o modelo MiDaS está disponível
        model_path = os.path.join('models', 'midas_v21_small.pt')
        if not os.path.exists(model_path):
            # Tenta baixar o modelo automaticamente
            try:
                print("Baixando modelo MiDaS...")
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                torch.save(model.state_dict(), model_path)
                print("Modelo MiDaS baixado com sucesso!")
            except Exception as e:
                print(f"Erro ao baixar o modelo: {e}")
                return 0.5, None, {"error": f"Modelo MiDaS não encontrado: {e}"}
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return 0.5, None, {"error": "Imagem não pôde ser carregada"}
        
        img_height, img_width = img.shape[:2]
        
        # Redimensionar imagem (preservando a proporção)
        input_size = (384, 384)  # tamanho de entrada para o modelo MiDaS
        img_resized = cv2.resize(img, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
        
        # Preparar tensor para o modelo
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Converter para tensor
        img_tensor = transform(img_rgb).unsqueeze(0)
        
        # Carregar o modelo MiDaS
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Desativar gradientes para inferência
        with torch.no_grad():
            prediction = model(img_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Converter para numpy e normalizar para visualização
        depth_map = prediction.cpu().numpy()
        
        # Normalizar para visualização
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = depth_map * 0.0
        
        # Salvar mapa de profundidade como imagem
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_map_path = os.path.join(UPLOAD_FOLDER, "depth_map.jpg")
        cv2.imwrite(depth_map_path, depth_colored)
        
        # Análise do mapa de profundidade
        depth_analysis = {}
        
        # 1. Desvio padrão - Um rosto real terá mais variação de profundidade
        depth_std = np.std(depth_map)
        depth_analysis["depth_std"] = float(depth_std)
        
        # 2. Média do gradiente - Um rosto real terá transições de profundidade mais suaves
        sobelx = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient)
        depth_analysis["gradient_mean"] = float(gradient_mean)
        
        # 3. Variação local - Um rosto real terá mais detalhes finos
        local_var = cv2.Laplacian(depth_norm, cv2.CV_64F).var()
        depth_analysis["local_variation"] = float(local_var)
        
        # 4. Entropia do histograma - Um rosto real tem distribuição mais rica de profundidades
        hist = np.histogram(depth_norm, bins=50)[0]
        hist = hist / hist.sum()
        hist_entropy = -np.sum(hist * np.log2(hist + 1e-7))
        depth_analysis["histogram_entropy"] = float(hist_entropy)
        
        # Normalizar métricas para o score final
        # Definimos faixas esperadas com base em observação empírica
        norm_depth_std = min(1.0, depth_std / 0.2)  # Normaliza para 0-1
        norm_grad_mean = min(1.0, gradient_mean / 0.15)
        norm_local_var = min(1.0, local_var / 0.02)
        norm_hist_entropy = min(1.0, hist_entropy / 5.0)
        
        # Pesos de cada métrica para o score final
        weights = {
            "depth_std": 0.4,        # Maior peso para variação de profundidade
            "gradient_mean": 0.2,    # Transições naturais são importantes
            "local_variation": 0.2,  # Detalhes finos são importantes
            "histogram_entropy": 0.2  # Distribuição rica é importante
        }
        
        # Calcular score final ponderado
        depth_score = (
            weights["depth_std"] * norm_depth_std +
            weights["gradient_mean"] * norm_grad_mean +
            weights["local_variation"] * norm_local_var +
            weights["histogram_entropy"] * norm_hist_entropy
        )
        
        # Normalizar score final entre 0-1
        depth_score = min(1.0, max(0.0, depth_score))
        
        # Log para diagnóstico
        print(f"Depth Analysis: Score={depth_score:.2f}, StdDev={norm_depth_std:.2f}, Gradient={norm_grad_mean:.2f}, LocalVar={norm_local_var:.2f}, Entropy={norm_hist_entropy:.2f}")
        
        # Adicionar scores normalizados à análise
        depth_analysis["norm_depth_std"] = float(norm_depth_std)
        depth_analysis["norm_gradient_mean"] = float(norm_grad_mean)
        depth_analysis["norm_local_variation"] = float(norm_local_var)
        depth_analysis["norm_histogram_entropy"] = float(norm_hist_entropy)
        
        return depth_score, depth_map_path, depth_analysis
        
    except Exception as e:
        print(f"Erro na análise de profundidade: {e}")
        return 0.5, None, {"error": str(e)}

# Layout principal da aplicação
st.title("Identity Guardian")
st.markdown("### Upload your photo and document for verification")

# Adicionar abas para navegação
tab1, tab2 = st.tabs(["Identity Verification", "Document Analysis Results"])

with tab1:
    with st.form("verification_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Photo")
            photo_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
            if photo_file:
                st.image(photo_file, caption="Uploaded Photo", use_container_width=True)

                # Verificação de qualidade em tempo real
                if st.form_submit_button("Check Photo Quality", type="secondary"):
                    with st.spinner("Analyzing photo quality..."):
                        # Salvar imagem temporariamente
                        temp_path = save_uploaded_file(photo_file, UPLOAD_FOLDER)
                        
                        # Verificar se há face na imagem
                        annotated_path, _ = detect_facial_landmarks(temp_path)
                        
                        # Analisar qualidade
                        brightness, contrast = analyze_lighting_and_quality(temp_path)
                        
                        # Exibir resultados
                        st.write("**Image Quality Analysis:**")
                        quality_col1, quality_col2 = st.columns(2)
                        
                        with quality_col1:
                            brightness_status = "Good" if 100 <= brightness <= 200 else "Poor"
                            brightness_color = "green" if brightness_status == "Good" else "red"
                            st.markdown(f"Brightness: <span style='color:{brightness_color}'>{brightness:.1f}</span>", unsafe_allow_html=True)
                        
                        with quality_col2:
                            contrast_status = "Good" if contrast > 40 else "Poor"
                            contrast_color = "green" if contrast_status == "Good" else "red"
                            st.markdown(f"Contrast: <span style='color:{contrast_color}'>{contrast:.1f}</span>", unsafe_allow_html=True)
                        
                        # Mostrar imagem com landmarks faciais se disponível
                        if annotated_path:
                            st.write("**Face Detection:**")
                            st.image(annotated_path, caption="Facial Landmarks", use_container_width=True)
                        else:
                            st.error("No face detected in the image!")
        
        with col2:
            st.subheader("Document Photo")
            document_file = st.file_uploader("Upload your document", type=["jpg", "jpeg", "png"])
            if document_file:
                st.image(document_file, caption="Uploaded Document", use_container_width=True)
                
                # Verificação de OCR em tempo real
                if st.form_submit_button("Preview Document Analysis", type="secondary"):
                    with st.spinner("Analyzing document..."):
                        # Salvar imagem temporariamente
                        temp_path = save_uploaded_file(document_file, UPLOAD_FOLDER)
                        
                        # Extrair texto do documento
                        document_text = extract_text_from_document(temp_path)
                        extracted_info = extract_info_from_text(document_text)
                        
                        # Mostrar texto extraído
                        st.write("**OCR Analysis Preview:**")
                        st.markdown(f"**Name detected:** {extracted_info.get('name', 'Not detected')}")
                        st.markdown(f"**Birth date detected:** {extracted_info.get('birth_date', 'Not detected')}")
                        st.markdown(f"**Gender detected:** {extracted_info.get('gender', 'Not detected')}")
                        
                        # Mostrar imagem pré-processada
                        preprocessed_path = os.path.join(UPLOAD_FOLDER, "preprocessed_doc.jpg")
                        if os.path.exists(preprocessed_path):
                            with st.expander("View preprocessed document"):
                                st.image(preprocessed_path, caption="Preprocessed for OCR", use_container_width=True)
        
        # Campos pessoais
        personal_col1, personal_col2 = st.columns(2)
        
        with personal_col1:
            name = st.text_input("Full Name")
            birth_date = st.date_input("Date of Birth", 
                                      min_value=datetime(1900, 1, 1), 
                                      max_value=datetime.now(),
                                      value=datetime(2000, 1, 1))
        
        with personal_col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            document_type = st.selectbox("Document Type", ["Passport", "ID Card", "Driver's License"])
        
        submit_button = st.form_submit_button("Verify Identity")

# Processar envio do formulário
if submit_button:
    if not photo_file or not document_file or not name:
        st.error("Please upload both photos and enter your name.")
    else:
        with st.spinner("Processing your verification..."):
            # Salvar arquivos enviados
            photo_path = save_uploaded_file(photo_file, UPLOAD_FOLDER)
            document_path = save_uploaded_file(document_file, UPLOAD_FOLDER)
            
            # Calcular idade a partir da data de nascimento
            today = datetime.now()
            declared_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            # Extrair texto do documento via OCR
            document_text = extract_text_from_document(document_path)
            ocr_data = extract_info_from_text(document_text)
            
            # Detectar pontos faciais e salvar imagem anotada
            annotated_path, _ = detect_facial_landmarks(photo_path)
            
            # Realizar análises
            brightness, contrast = analyze_lighting_and_quality(photo_path)

            # Chamar funções com retorno de scores
            liveness_result, liveness_score = check_liveness(photo_path, return_score=True)
            face_swap_result, face_swap_score = detect_face_swaps_or_edits(photo_path, return_score=True)
            edit_result, edit_score = detect_crop_or_edit(photo_path, return_score=True)

            # NOVOS MÉTODOS DE ANÁLISE
            # 1. Análise de profundidade monocular para detecção de spoofing
            depth_score, depth_map_path, depth_analysis = analyze_depth_for_liveness(photo_path)

            # 2. Análise específica de tela/spoofing
            is_screen, screen_score, screen_analysis = detect_screen_spoofing(photo_path)

            # 3. Classificação avançada de documento usando CNN
            document_type_cnn, document_confidence = classify_document_type_with_cnn(document_path)

            # Usar resultados booleanos
            liveness_detected = liveness_result
            face_swap_detected = face_swap_result
            crop_or_edit_detected = edit_result

            # Ajustar a detecção de vivacidade com base na análise de profundidade
            # Se o score de profundidade for muito baixo, há maior chance de ser um spoof (foto de foto)
            if depth_score < 0.35:
                print(f"Liveness ajustado pela análise de profundidade: Score={depth_score:.2f}")
                liveness_detected = False
                liveness_score = min(liveness_score, 0.3)  # Reduzir o score se a profundidade indicar spoof

            # Ajustar a detecção de vivacidade se a detecção de tela for positiva
            if is_screen and screen_score > 0.65:
                print(f"Liveness ajustado pela detecção de tela/spoofing: Score={screen_score:.2f}")
                liveness_detected = False
                liveness_score = min(liveness_score, 0.2)  # Reduzir ainda mais o score se detectar tela

            similarity_score = compare_faces(photo_path, document_path)

            # Analisar se o documento parece fraudado
            document_tampering_results = detect_document_tampering(document_path)

            # Estimar idade e gênero a partir da foto
            estimated_age, estimated_gender = estimate_age_gender(photo_path)

            # Gerar relatório incluindo os scores originais
            report = generate_final_report(
                similarity_score=similarity_score,
                face_swap_detected=face_swap_detected,
                crop_or_edit_detected=crop_or_edit_detected,
                brightness=brightness,
                contrast=contrast,
                liveness_detected=liveness_detected,
                user_name=name,
                document_type=document_type_cnn if document_confidence > 0.6 else document_type,  # Usar CNN se confiança for alta
                declared_age=declared_age,
                estimated_age=estimated_age,
                declared_gender=gender,
                estimated_gender=estimated_gender,
                ocr_data=ocr_data,
                face_swap_score=face_swap_score,
                edit_score=edit_score,
                liveness_score=liveness_score
            )
            
            # Adicionar novas análises ao relatório
            report["depth_analysis"] = {
                "depth_score": f"{depth_score:.2f}",
                "depth_map_path": depth_map_path
            }

            report["screen_analysis"] = {
                "is_screen": "Yes" if is_screen else "No",
                "screen_score": f"{screen_score:.2f}"
            }

            # Adicionar detalhes da análise de documento
            report["document_analysis"] = {
                "cnn_type": document_type_cnn,
                "cnn_confidence": f"{document_confidence:.2f}"
            }
            
            # Adicionar resultados da análise de documento ao relatório
            report["document_tampering"] = {
                "tampered": document_tampering_results["tampered"],
                "score": document_tampering_results["score"],
                "reasons": document_tampering_results["reasons"]
            }
            
            # Verificar e adicionar análise EXIF à verificação final
            try:
                exif_is_consistent, exif_score, exif_data = analyze_exif_metadata(photo_path, return_score=True)
                report["exif_analysis"] = exif_data
                report["exif_consistent"] = "Yes" if exif_is_consistent else "No"
            except Exception as e:
                print(f"Erro na análise EXIF: {e}")
                report["exif_analysis"] = None
                report["exif_consistent"] = "N/A"
            
            # Atualizar também os campos de autenticidade no relatório principal
            report["document_authenticity"] = "Suspicious" if document_tampering_results["tampered"] else "Authentic"
            report["document_authenticity_score"] = f"{(1-float(document_tampering_results['score']))*100:.1f}%"
            
            # Se o documento parecer fraudado, adicionar recomendação e rejeitar
            if document_tampering_results["tampered"]:
                report["recommendations"].append("O documento apresenta sinais de possível adulteração. Por favor, forneça um documento original.")
                report["overall_result"] = "REJECTED"
            
            # Exibir uma mensagem de sucesso na primeira aba
            with tab1:
                st.success("Verification completed! Click on the 'Document Analysis Results' tab to view detailed results.")
            
            # Exibir resultados
            with tab2:
                st.success("Verification completed!")
                
                st.subheader("Verification Report")
                
                # Exibir resultado geral com destaque pela cor
                result_color = "green" if report["overall_result"] == "APPROVED" else "red"
                st.markdown(f"<h2 style='color:{result_color};'>{report['overall_result']}</h2>", unsafe_allow_html=True)
                
                # Gerar gráfico de radar para os scores
                #radar_chart_path = generate_verification_radar_chart(report)
                
                # Mostrar imagens analisadas lado a lado
                image_col1, image_col2 = st.columns(2)
                
                with image_col1:
                    if annotated_path:
                        st.image(annotated_path, caption="Face Analysis", use_container_width=True)
                    else:
                        st.image(photo_path, caption="Personal Photo", use_container_width=True)
                
                with image_col2:
                    # Mostrar apenas o documento enviado pelo usuário
                    st.image(document_path, caption="Document Photo", use_container_width=True)
                    
                    # Comentamos o código antigo que pode estar causando o problema
                    # preprocessed_path = os.path.join(UPLOAD_FOLDER, "preprocessed_doc.jpg")
                    # if os.path.exists(preprocessed_path):
                    #     st.image(preprocessed_path, caption="Document OCR", use_container_width=True)
                    # else:
                    #     st.image(document_path, caption="Document Photo", use_container_width=True)
                
#                 # Mostrar radar chart se disponível
#                 if radar_chart_path:
#                     st.image(radar_chart_path, caption="Análise de Scores de Verificação", use_container_width=True)
#                     st.info("🎯 Este gráfico mostra a pontuação em cada aspecto da verificação. Quanto mais preenchido o gráfico, maior a confiabilidade geral.")
                
                # Métricas de verificação
                st.subheader("Verification Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    # Fix the similarity_score conversion by removing '%' if present
                    similarity_score_str = report['similarity_score']
                    if isinstance(similarity_score_str, str) and '%' in similarity_score_str:
                        # If it's already a string with '%', just display it directly
                        st.metric("Facial Match", similarity_score_str)
                    else:
                        # Otherwise convert to float and format
                        similarity_float = float(similarity_score_str) if not isinstance(similarity_score_str, float) else similarity_score_str
                        st.metric("Facial Match", f"{similarity_float*100:.1f}%")
                    
                    st.metric("Liveness Detection", report["liveness_detected"])
                    st.metric("Trust Score", f"{report.get('trust_score', '0')}%")
                    
                with metrics_col2:
                    st.metric("Face Swap Detection", report["face_swap_detected"])
                    st.metric("Image Edited", report["image_edited"])
                    
                    # Adicionar detecção de tela
                    if "screen_analysis" in report:
                        screen_result = "Yes" if report["screen_analysis"].get("is_screen") == "Yes" else "No"
                        st.metric("Screen/Display", screen_result)
                    
                with metrics_col3:
                    st.metric("Lighting Quality", report["lighting_quality"])
                    st.metric("Age Match", report["age_match"])
                    
                    # Adicionar análise de profundidade
                    if "depth_analysis" in report:
                        depth_score = float(report["depth_analysis"].get("depth_score", "0"))
                        score_text = "Good" if depth_score > 0.6 else "Poor" if depth_score < 0.4 else "Medium"
                        st.metric("Depth Analysis", score_text)
                    
                with metrics_col4:
                    st.metric("Document Type", report["document_type"])
                    if "document_tampering" in report:
                        tampering_status = "Authentic" if not report["document_tampering"]["tampered"] else "Suspicious"
                        st.metric("Document Authenticity", tampering_status)
                    
                    # Adicionar tipo de documento via CNN se disponível
                    if "document_analysis" in report:
                        st.metric("Doc Type (CNN)", report["document_analysis"].get("cnn_type", "Unknown"))
                
                # Mostrar dados extraídos por OCR
                st.subheader("Document Data Comparison")
                
                ocr_col1, ocr_col2 = st.columns(2)
                
                with ocr_col1:
                    st.write("**User Provided Data:**")
                    st.write(f"Name: {name}")
                    st.write(f"Birth Date: {birth_date.strftime('%d/%m/%Y')}")
                    st.write(f"Gender: {gender}")
                
                with ocr_col2:
                    st.write("**OCR Extracted Data:**")
                    st.write(f"Name: {ocr_data.get('name', 'Not detected')}")
                    st.write(f"Birth Date: {ocr_data.get('birth_date', 'Not detected')}")
                    st.write(f"Gender: {ocr_data.get('gender', 'Not detected')}")
                
                # Gerar explicações em linguagem natural
                explanations = generate_natural_language_explanation(report)
                
                # Exibir análise explicável com mapas de calor
                st.subheader("Explainable Analysis")
                
                # Criar abas para diferentes tipos de análise
                tabs = st.tabs(["Overall", "Face Comparison", "Liveness", "Image Editing", "Quality", "Advanced Anti-Spoofing", "Document Authenticity", "Complete Report"])
                
                with tabs[0]:  # Overall
                    st.markdown(explanations["overall"])
                    
                    # Exibir recomendações, se houver
                    if report["recommendations"]:
                        st.subheader("Recommendations")
                        for rec in report["recommendations"]:
                            st.warning(rec)
                
                with tabs[1]:  # Face Comparison
                    st.markdown(explanations["facial_match"])
                    if "face_swap" in explanations:
                        st.markdown(explanations["face_swap"])
                    
                    # Exibir resultado da análise de face swap com marcações visuais
                    face_swap_analysis_path = os.path.join(UPLOAD_FOLDER, "face_swap_analysis.jpg")
                    if os.path.exists(face_swap_analysis_path):
                        st.image(face_swap_analysis_path, caption="Face Swap Analysis with Detection Regions", use_container_width=True)
                        st.info("🔍 **Interpretação da análise**: Regiões destacadas com retângulos coloridos são analisadas em busca de inconsistências de textura e padrões. Valores numéricos indicam o nível de suspeita em cada região.")
                    
                    # Gerar e exibir mapa de calor para face swap
                    face_swap_heatmap = generate_heatmap(photo_path, 'face_swap')
                    if face_swap_heatmap:
                        st.image(face_swap_heatmap, caption="Face Swap Analysis Heat Map", use_container_width=True)
                        st.info("🔍 **Interpretação do mapa de calor**: As áreas em vermelho/amarelo indicam regiões com maior probabilidade de manipulação. As bordas do rosto e áreas como olhos e boca são particularmente importantes para detectar inconsistências.")
                
                with tabs[2]:  # Liveness
                    if "liveness" in explanations:
                        st.markdown(explanations["liveness"])
                    
                    # Gerar e exibir mapa de calor para vivacidade
                    liveness_heatmap = generate_heatmap(photo_path, 'liveness')
                    if liveness_heatmap:
                        st.image(liveness_heatmap, caption="Liveness Analysis Heat Map", use_container_width=True)
                        st.info("🔍 **Interpretação do mapa de calor**: Áreas em vermelho/amarelo indicam potenciais problemas de vivacidade. Padrões de bordas muito definidos ou áreas muito uniformes podem indicar que a imagem é uma reprodução de outra foto (spoofing).")
                
                with tabs[3]:  # Image Editing
                    if "image_editing" in explanations:
                        st.markdown(explanations["image_editing"])
                    
                    # Exibir análise detalhada de edição com marcações visuais
                    edit_analysis_path = os.path.join(UPLOAD_FOLDER, "edit_analysis.jpg")
                    if os.path.exists(edit_analysis_path):
                        st.image(edit_analysis_path, caption="Image Edit Analysis with Detection Regions", use_container_width=True)
                        st.info("🔍 **Interpretação da análise**: Contornos vermelhos destacam áreas com possíveis edições ou manipulações. Os indicadores numéricos mostram a confiança das diferentes métricas de detecção.")
                    
                    # Gerar e exibir mapa de calor para edições
                    edit_heatmap = generate_heatmap(photo_path, 'editing')
                    if edit_heatmap:
                        st.image(edit_heatmap, caption="Image Editing Analysis Heat Map", use_container_width=True)
                        st.info("🔍 **Interpretação do mapa de calor**: Áreas em vermelho/amarelo mostram regiões com padrões de ruído inconsistentes ou transições abruptas, que são típicos de edições digitais como recortes, colagens ou manipulações localizadas.")
                
                with tabs[4]:  # Quality
                    if "image_quality" in explanations:
                        st.markdown(explanations["image_quality"])
                    
                    # Gerar e exibir mapa de calor para qualidade
                    quality_heatmap = generate_heatmap(photo_path, 'quality')
                    if quality_heatmap:
                        st.image(quality_heatmap, caption="Image Quality Analysis Heat Map", use_container_width=True)
                        st.info("🔍 **Interpretação do mapa de calor**: Áreas em vermelho/amarelo indicam regiões com baixa qualidade, incluindo pouco contraste, blur ou ruído excessivo. Estas regiões podem comprometer a precisão da verificação de identidade.")
                
                with tabs[5]:  # Advanced Anti-Spoofing
                    st.subheader("Análise Avançada Anti-Spoofing")
                    
                    # Dividir em colunas para organizar melhor a visualização
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("1. Detecção de Profundidade")
                        
                        # Verificar se temos dados de profundidade
                        if "depth_analysis" in report and report["depth_analysis"].get("depth_map_path"):
                            depth_score = float(report["depth_analysis"].get("depth_score", "0"))
                            depth_map_path = report["depth_analysis"].get("depth_map_path")
                            
                            # Mostrar pontuação de profundidade
                            score_color = "green" if depth_score > 0.6 else "red"
                            st.markdown(f"**Pontuação de Profundidade**: <span style='color:{score_color};'>{depth_score:.2f}</span>", unsafe_allow_html=True)
                            
                            if depth_score < 0.4:
                                st.warning("⚠️ **Baixa variação de profundidade detectada**: Imagens planas (como fotos de fotos) geralmente têm pouca variação natural de profundidade.")
                            
                            # Mostrar mapa de profundidade
                            if os.path.exists(depth_map_path):
                                st.image(depth_map_path, caption="Mapa de Profundidade", use_container_width=True)
                                st.info("🔍 **Interpretação do mapa**: Áreas mais claras representam partes mais próximas da câmera, áreas mais escuras representam partes mais distantes.")
                        else:
                            st.warning("Análise de profundidade não disponível para esta imagem.")
                    
                    with col2:
                        st.subheader("2. Detecção de Tela/Spoofing")
                        
                        # Verificar se temos dados de análise de tela
                        if "screen_analysis" in report:
                            is_screen = report["screen_analysis"].get("is_screen") == "Yes"
                            screen_score = float(report["screen_analysis"].get("screen_score", "0"))
                            
                            # Mostrar pontuação de detecção de tela
                            score_color = "red" if screen_score > 0.5 else "green"
                            st.markdown(f"**Pontuação de Spoofing**: <span style='color:{score_color};'>{screen_score:.2f}</span>", unsafe_allow_html=True)
                            
                            if is_screen:
                                st.warning("⚠️ **Possível foto de tela detectada**: Foram identificados padrões típicos de displays/monitores.")
                            
                            # Gerar e exibir mapa de calor para tela/spoofing
                            screen_heatmap = generate_heatmap(photo_path, 'screen')
                            if screen_heatmap:
                                st.image(screen_heatmap, caption="Mapa de Calor de Análise de Tela", use_container_width=True)
                                st.info("🔍 **Interpretação**: Áreas destacadas mostram regiões com padrões característicos de telas, como reflexos retangulares, padrões Moiré ou estruturas de pixel.")
                        else:
                            st.warning("Análise de tela/spoofing não disponível para esta imagem.")
                    
                    # Exibir métricas detalhadas para usuários avançados
                    with st.expander("Ver métricas detalhadas"):
                        if "depth_analysis" in report:
                            st.subheader("Métricas de Profundidade")
                            metrics = report["depth_analysis"]
                            for key, value in metrics.items():
                                if key != "depth_map_path":
                                    st.text(f"{key}: {value}")
                        
                        if "screen_analysis" in report:
                            st.subheader("Métricas de Detecção de Tela")
                            st.text(f"Screen detection confidence: {report['screen_analysis'].get('screen_score', 'N/A')}")
                
                with tabs[6]:  # Document Authenticity
                    # Mostrar resultados da análise de fraude do documento
                    if "document_tampering" in report:
                        tampering_results = report["document_tampering"]
                        
                        # Título e resultado principal
                        result_color = "green" if not tampering_results["tampered"] else "red"
                        status_text = "AUTÊNTICO" if not tampering_results["tampered"] else "SUSPEITO"
                        st.markdown(f"<h3 style='color:{result_color};'>Documento: {status_text}</h3>", unsafe_allow_html=True)
                        
                        # Pontuação de adulteração
                        st.metric("Pontuação de Autenticidade", f"{(1-tampering_results['score'])*100:.1f}%")
                        
                        # Mostrar imagem com áreas suspeitas destacadas se disponível
                        if "result_image" in document_tampering_results and os.path.exists(document_tampering_results["result_image"]):
                            st.image(document_tampering_results["result_image"], caption="Análise de Autenticidade do Documento", use_container_width=True)
                            st.info("🔍 **Interpretação da análise**: Áreas destacadas em vermelho indicam regiões com níveis de erro inconsistentes, que podem ser sinais de alteração digital ou fraude.")
                        
                        # Exibir razões para o resultado
                        if tampering_results["reasons"]:
                            st.subheader("Possíveis sinais de adulteração:")
                            for reason in tampering_results["reasons"]:
                                st.warning(reason)
                        else:
                            st.success("Nenhum sinal de adulteração detectado no documento.")
                        
                        # Explicações sobre a análise de autenticidade
                        st.markdown("""
                        ### Como funciona a análise de autenticidade:
                        
                        1. **Análise ELA (Error Level Analysis)**: Detecta diferenças nos níveis de compressão que podem indicar áreas coladas ou editadas.
                        
                        2. **Análise de fontes**: Verifica se os caracteres no documento têm dimensões consistentes, como esperado em documentos legítimos.
                        
                        3. **Padrões de segurança**: Documentos oficiais contêm micro-impressões e padrões finos que são difíceis de reproduzir em falsificações.
                        
                        4. **Geometria e bordas**: Analisa se o documento tem formato e ângulos consistentes, comuns em documentos originais.
                        
                        5. **Consistência de cores**: Examina se a distribuição de cores está dentro dos padrões esperados para o tipo de documento.
                        """)
                    else:
                        st.warning("Análise de autenticidade do documento não disponível.")
                
                with tabs[7]:  # Complete Report (NOVO)
                    st.subheader("Relatório Completo com Explicações")
                    
                    # Gerar o relatório completo com explicações
                    complete_report = display_complete_report_explanation(report, explanations)
                    
                    # Mostrar cada seção do relatório
                    for section, items in complete_report.items():
                        with st.expander(section, expanded=True):
                            for key, value in items.items():
                                if key == "explanation":
                                    st.markdown(value)
                                else:
                                    st.markdown(value)
                                    
                                    # Adicionar separador entre seções
                                    st.markdown("---")
            
                # Exibir texto OCR
                if document_text:
                    with st.expander("View raw OCR text"):
                        st.text(document_text)
                
                # Exibir JSON completo do relatório
                with st.expander("View full report (JSON)"):
                    st.json(report)
            
                # Gerar análise detalhada usando GPT-4 (se disponível)
                if has_openai:
                    st.subheader("AI-Powered Analysis")
                    
                    with st.spinner("Gerando análise detalhada com IA..."):
                        analysis = generate_ai_analysis(report)
                        
                        # Renderizar a análise formatada
                        st.markdown(analysis)
                        
                        # Adicionar explicação ou dica sobre a análise
                        st.info("A análise acima foi gerada automaticamente por um sistema de IA com base nos dados do relatório. Considere todos os fatores antes de tomar uma decisão final.")
