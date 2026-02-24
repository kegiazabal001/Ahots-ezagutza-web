# Euskara Transcription App

Aplicación de transcripción de voz a texto en euskera (Basque) utilizando faster-whisper con el modelo `xezpeleta/whisper-medium-eu-ct2`.

## Características

- 🎤 **Transcripción en tiempo real** vía micrófono (WebSocket)
- 📁 **Transcripción de archivos** de audio (upload)
- 🔍 **Detección de actividad de voz** (VAD) para segmentar audio
- 🔇 **Reducción de ruido** automática con `noisereduce` y normalización de volumen
- 🐳 **Dockerizado** para fácil despliegue
- 🔒 **HTTPS** para acceso al micrófono del navegador

## Inicio Rápido con Docker (Recomendado)

### Prerrequisitos

- Docker y Docker Compose instalados
- Certificados SSL (`cert.pem` y `key.pem`) para HTTPS

### Ejecutar

```bash
# Construir y ejecutar
docker compose up -d

# Ver logs
docker compose logs -f

# Detener
docker compose down
```

La aplicación estará disponible en `https://localhost:8000`

**Nota:** La primera ejecución descargará el modelo Whisper (~1.5GB). Se guarda en un volumen de Docker para ejecuciones posteriores.

## Instalación Local (Sin Docker)

### Prerrequisitos

- Python 3.13+
- ffmpeg instalado en el sistema
- Certificados SSL (`cert.pem` y `key.pem`)

### Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor web
python app.py
```

La aplicación estará disponible en `https://localhost:8000`

### CLI para transcribir archivos

```bash
# Editar AUDIO_FILE en whisper-app.py con la ruta de tu archivo
python whisper-app.py
```

## Generación de Certificados SSL

Para desarrollo local, puedes generar certificados auto-firmados:

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

**Nota:** Los navegadores mostrarán una advertencia de seguridad con certificados auto-firmados. Para producción, usa certificados de Let's Encrypt o similar.

## Arquitectura

- **Backend:** FastAPI + faster-whisper
- **Frontend:** HTML/JavaScript vanilla con Web Audio API
- **VAD:** Silero Voice Activity Detection (threshold=0.15)
- **Mejora de audio:** noisereduce (reducción espectral) + normalización RMS
- **Modelo:** xezpeleta/whisper-medium-eu-ct2 (optimizado para euskera)
- **Audio:** Conversión a 16kHz mono vía ffmpeg

## Endpoints

- `GET /` - Interfaz web
- `POST /api/transcribe` - Transcripción de archivo completa (retorna JSON con todos los segmentos)
- `POST /api/transcribe-stream` - Transcripción de archivo con progreso en tiempo real (Server-Sent Events)
- `WS /ws/transcribe` - Transcripción en tiempo real vía micrófono (WebSocket)

## Rendimiento de Docker

Docker tiene un overhead mínimo (<5%) para cargas CPU-intensivas como esta. El modelo se cachea en un volumen de Docker para evitar re-descargas.

### Optimizaciones

Puedes ajustar los límites de recursos en `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

## Licencia

Este proyecto usa el modelo [xezpeleta/whisper-medium-eu-ct2](https://huggingface.co/xezpeleta/whisper-medium-eu-ct2) de Hugging Face.
