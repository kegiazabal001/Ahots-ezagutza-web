# Euskara Transcription App

Aplicación web de transcripción de voz a texto en euskera usando [faster-whisper](https://github.com/SYSTRAN/faster-whisper) con el modelo [`xezpeleta/whisper-medium-eu-ct2`](https://huggingface.co/xezpeleta/whisper-medium-eu-ct2).

## Características

- **Transcripción en tiempo real** vía micrófono (WebSocket con sliding window)
- **Transcripción de archivos** de audio con progreso en tiempo real (Server-Sent Events)
- **Detección de actividad de voz** (VAD) con Silero para segmentar el audio
- **Mejora de audio** automática: reducción de ruido espectral (`noisereduce`) y normalización RMS
- **Optimizado para voz susurrada** gracias a parámetros VAD ajustados y amplificación adaptativa
- **Dockerizado** para despliegue rápido

## Inicio rápido con Docker

### Prerrequisitos

- Docker y Docker Compose
- Certificados SSL (`cert.pem` y `key.pem`) — necesarios para que el navegador acceda al micrófono

```bash
# Generar certificados auto-firmados para desarrollo
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

### Ejecutar

```bash
docker compose up -d

# Ver logs
docker compose logs -f

# Detener
docker compose down
```

La aplicación estará disponible en **`https://localhost:8050`**

> El modelo Whisper (~1.5 GB) se descarga automáticamente en la primera ejecución y se guarda en un volumen de Docker (`whisper-models`) para no volver a descargarlo.

## Instalación local

### Prerrequisitos

- Python 3.13+
- `ffmpeg` instalado en el sistema
- Certificados SSL (`cert.pem` y `key.pem`)

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python app.py
```

La aplicación estará disponible en **`https://localhost:8000`**

### CLI para transcribir archivos

```bash
# Editar la variable AUDIO_FILE en whisper-app.py con la ruta de tu archivo
python whisper-app.py
```

## Arquitectura

| Componente | Tecnología |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | HTML/JS vanilla + Web Audio API |
| Modelo ASR | faster-whisper (`xezpeleta/whisper-medium-eu-ct2`) |
| VAD | Silero (threshold=0.15) |
| Mejora de audio | noisereduce (sustracción espectral) + normalización RMS |
| Conversión de audio | ffmpeg (→ 16 kHz mono) |
| Cómputo | CPU, `int8` quantization |

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Interfaz web |
| `POST` | `/api/transcribe` | Transcripción completa de un archivo (JSON) |
| `POST` | `/api/transcribe-stream` | Transcripción de archivo con progreso (SSE) |
| `WS` | `/ws/transcribe` | Transcripción en tiempo real por micrófono |

## Configuración de recursos (Docker)

Para ajustar los límites de CPU/memoria, descomenta y edita en `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

## Licencia

Modelo: [xezpeleta/whisper-medium-eu-ct2](https://huggingface.co/xezpeleta/whisper-medium-eu-ct2) — Hugging Face.
