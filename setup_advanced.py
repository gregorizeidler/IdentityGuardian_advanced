import os
import sys
import subprocess

def install_packages():
    """
    Instala os pacotes necessários para as funcionalidades avançadas
    """
    print("Instalando pacotes para funcionalidades avançadas...")
    
    # Lista de pacotes
    packages = [
        "torch",
        "torchvision",
        "matplotlib==3.8.2",
        "scikit-image==0.22.0"
    ]
    
    # Instalar pacotes
    for package in packages:
        print(f"Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Pacotes instalados com sucesso!")

def download_models():
    """
    Faz download dos modelos pré-treinados necessários
    """
    print("Criando diretório para modelos...")
    os.makedirs("models", exist_ok=True)
    
    # Verificar se precisamos baixar o modelo MiDaS
    if not os.path.exists("models/midas_v21_small.pt"):
        print("Baixando modelo MiDaS (isto pode levar alguns minutos)...")
        
        try:
            # Usando PyTorch Hub
            import torch
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            torch.save(model.state_dict(), "models/midas_v21_small.pt")
            print("Modelo MiDaS baixado com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar modelo MiDaS: {e}")
            print("Você precisará baixar manualmente ou usar o aplicativo sem o recurso de análise de profundidade.")
    else:
        print("Modelo MiDaS já existe.")
        
    print("Configuração concluída!")

if __name__ == "__main__":
    print("=== Configuração de Recursos Avançados do Identity Guardian ===")
    print("Este script instalará pacotes e modelos necessários para as funcionalidades avançadas.")
    
    try:
        install_packages()
        download_models()
        
        print("\nConfigurações avançadas concluídas com sucesso!")
        print("Agora você pode executar 'python -m streamlit run app.py' para iniciar o aplicativo com todos os recursos.")
        
    except Exception as e:
        print(f"\nErro durante a configuração: {e}")
        print("Você ainda pode executar o aplicativo, mas algumas funcionalidades avançadas podem não estar disponíveis.") 
