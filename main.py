from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#from contextlib import asynccontextmanager
#import whisper
import tempfile
import os
from pathlib import Path
import sys

app = FastAPI(title="Audio Transcription API")

# Variable global para el modelo
#model = None

#@asynccontextmanager
#async def lifespan(app: FastAPI):
    # Código que se ejecuta AL INICIAR
    #global model
    #print("=" * 50)
    #print("🚀 Iniciando WhisperAPI...")
    
    # Determinar la ruta base
    #if getattr(sys, 'frozen', False):
        #BASE_DIR = Path(sys._MEIPASS)
        #EXEC_DIR = Path(sys.executable).parent
    #else:
     #   BASE_DIR = Path(__file__).parent
      #  EXEC_DIR = BASE_DIR

# Configurar la ruta de ffmpeg local
BASE_DIR = Path(__file__).parent
FFMPEG_PATH = BASE_DIR / "ffmpeg" / "bin"

# Agregar ffmpeg al PATH del sistema temporalmente
if FFMPEG_PATH.exists():
    os.environ["PATH"] = str(FFMPEG_PATH) + os.pathsep + os.environ["PATH"]
    print(f"FFmpeg configurado desde: {FFMPEG_PATH}")
else:
    print(f"⚠️ Advertencia: No se encontró ffmpeg en {FFMPEG_PATH}")
    print("El sistema intentará usar ffmpeg del PATH del sistema")

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo Whisper (small)
print("Cargando modelo Whisper...")

# Buscar modelo local primero
MODEL_PATH = BASE_DIR / "models" / "small.pt"

if MODEL_PATH.exists():
    print(f"Usando modelo local: {MODEL_PATH}")
    model = whisper.load_model("small", download_root=str(BASE_DIR / "models"))
else:
    print("Descargando modelo Whisper (esto puede tardar la primera vez)...")
    model = whisper.load_model("small")

print("Modelo cargado exitosamente")

@app.get("/")
async def root():
    return {
        "message": "API de Transcripción de Audio",
        "status": "active",
        "model": "whisper-small"
    }

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Endpoint para transcribir audio a texto usando Whisper
    """
    temp_file_path = None
    
    try:
        # Validar extensión de archivo
        print(f"Archivo recibido: {audio.filename}, tipo: {audio.content_type}")
        
        # Crear un archivo temporal para guardar el audio
        suffix = Path(audio.filename).suffix if audio.filename else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Leer y guardar el contenido del archivo
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"Archivo guardado temporalmente en: {temp_file_path}")
        print(f"Iniciando transcripción...")
        
        # Transcribir el audio (sin especificar idioma para detección automática)
        result = model.transcribe(
            temp_file_path,
            fp16=False,  # Forzar FP32 para evitar problemas en CPU
            language="spanish"  # Detección automática de idioma
        )
        
        print(f"Transcripción completada: {result['text'][:100]}...")
        
        # Eliminar el archivo temporal
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return {
            "success": True,
            "text": result["text"],
            "transcription": result["text"],
            "language": result.get("language", "unknown"),
            "filename": audio.filename
        }
    
    except Exception as e:
        # Asegurarse de eliminar el archivo temporal en caso de error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        print(f"Error completo: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error al transcribir el audio: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

#if __name__ == "__main__":
 #   import uvicorn
  #  uvicorn.run(app, host="0.0.0.0", port=10000, log_config=None)
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=10000)
