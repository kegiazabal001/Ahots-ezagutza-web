import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import tempfile
import os
from faster_whisper import WhisperModel


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.temp_file = None
        self.stream = None
        self.lock = threading.Lock()
        
        # Parámetros de audio
        self.RATE = 16000
        self.CHANNELS = 1
        
        # Modelo Whisper
        self.model = None
        self.model_loaded = False
    
    def initialize_audio(self):
        """Inicializa y valida dispositivos de audio"""
        try:
            # Verificar dispositivo de entrada por defecto
            default_input = sd.query_devices(kind='input')
            
            if default_input is None:
                return False, "No se encontró dispositivo de entrada de audio"
            
            # Verificar que el dispositivo soporte la tasa de muestreo
            try:
                sd.check_input_settings(
                    device=default_input['index'],
                    channels=self.CHANNELS,
                    samplerate=self.RATE
                )
            except sd.PortAudioError as e:
                return False, f"El dispositivo no soporta la configuración requerida: {str(e)}"
            
            return True, f"Dispositivo listo: {default_input['name']}"
            
        except Exception as e:
            return False, f"Error inicializando audio: {str(e)}"
    
    def load_model(self):
        """Carga el modelo Whisper"""
        try:
            self.model = WhisperModel(
                "xezpeleta/whisper-medium-eu-ct2", 
                device="cpu", 
                compute_type="int8"
            )
            self.model_loaded = True
            return True, "Modelo cargado correctamente"
        except Exception as e:
            return False, f"Error cargando modelo: {str(e)}"
    
    def audio_callback(self, indata, frames, time, status):
        """Callback para capturar audio en tiempo real"""
        if status:
            print(f"Audio status: {status}")
        
        with self.lock:
            if self.recording:
                self.frames.append(indata.copy())
    
    def start_recording(self):
        """Inicia la grabación de audio"""
        with self.lock:
            if self.recording:
                return False, "Ya hay una grabación en curso"
            
            self.recording = True
            self.frames = []
        
        try:
            # Crear e iniciar stream
            self.stream = sd.InputStream(
                samplerate=self.RATE,
                channels=self.CHANNELS,
                dtype='int16',
                callback=self.audio_callback
            )
            self.stream.start()
            
            return True, "Grabación iniciada"
            
        except Exception as e:
            with self.lock:
                self.recording = False
            return False, f"Error iniciando grabación: {str(e)}"
    
    def stop_recording(self):
        """Detiene la grabación y guarda el archivo"""
        with self.lock:
            if not self.recording:
                return None, "No hay grabación activa"
            self.recording = False
        
        # Detener y cerrar stream
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Error cerrando stream: {e}")
        
        # Verificar que hay datos
        if not self.frames:
            return None, "No se grabó ningún audio"
        
        try:
            # Concatenar todos los frames
            audio_data = np.concatenate(self.frames, axis=0)
            
            # Crear archivo temporal
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            # Guardar como WAV
            sf.write(
                self.temp_file.name, 
                audio_data, 
                self.RATE, 
                subtype='PCM_16'
            )
            
            return self.temp_file.name, "Audio guardado correctamente"
            
        except Exception as e:
            return None, f"Error guardando audio: {str(e)}"
    
    def transcribe(self, audio_file):
        """Transcribe el archivo de audio usando Whisper"""
        if not self.model_loaded:
            return None, "Error: Modelo no cargado"
        
        try:
            segments, info = self.model.transcribe(audio_file, language="eu")
            
            transcription = []
            for segment in segments:
                transcription.append(
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                )
            
            if not transcription:
                return "No se detectó audio o el audio está en silencio", "Advertencia"
            
            return "\n".join(transcription), "Transcripción exitosa"
            
        except Exception as e:
            return None, f"Error en transcripción: {str(e)}"
    
    def cleanup(self):
        """Limpia todos los recursos"""
        with self.lock:
            self.recording = False
        
        # Cerrar stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        # Eliminar archivo temporal
        if self.temp_file and os.path.exists(self.temp_file.name):
            try:
                os.unlink(self.temp_file.name)
            except:
                pass


class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grabadora con Whisper")
        self.root.geometry("600x400")
        
        self.recorder = AudioRecorder()
        self.is_recording = False
        
        # Crear interfaz
        self.create_widgets()
        
        # Inicializar en segundo plano
        self.initialize_async()
    
    def create_widgets(self):
        """Crea la interfaz gráfica"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Botón de grabación
        self.record_button = ttk.Button(
            main_frame, 
            text="🎤 Iniciar Grabación", 
            command=self.toggle_recording,
            state="disabled"
        )
        self.record_button.grid(row=0, column=0, pady=10)
        
        # Label de estado
        self.status_label = ttk.Label(
            main_frame, 
            text="Inicializando...", 
            foreground="blue"
        )
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Área de texto para transcripción
        ttk.Label(main_frame, text="Transcripción:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        
        self.text_box = scrolledtext.ScrolledText(
            main_frame, 
            width=70, 
            height=15, 
            wrap=tk.WORD
        )
        self.text_box.grid(row=3, column=0, pady=5)
        
        # Configurar expansión de la ventana
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
    
    def initialize_async(self):
        """Inicializa audio y modelo de forma asíncrona"""
        def initialize():
            # Inicializar audio
            success, msg = self.recorder.initialize_audio()
            if not success:
                self.root.after(0, lambda: self.show_init_error(msg))
                return
            
            # Actualizar estado
            self.root.after(0, lambda: self.update_status(
                f"Audio OK. Cargando modelo...", "blue"
            ))
            
            # Cargar modelo Whisper
            success, msg = self.recorder.load_model()
            if not success:
                self.root.after(0, lambda: self.show_init_error(msg))
                return
            
            # Todo listo
            self.root.after(0, self.enable_recording)
        
        threading.Thread(target=initialize, daemon=True).start()
    
    def show_init_error(self, error):
        """Muestra error de inicialización"""
        self.status_label.config(text=f"Error: {error}", foreground="red")
        messagebox.showerror("Error de inicialización", error)
    
    def update_status(self, text, color):
        """Actualiza el texto de estado"""
        self.status_label.config(text=text, foreground=color)
    
    def enable_recording(self):
        """Habilita el botón de grabación"""
        self.status_label.config(text="✓ Listo para grabar", foreground="green")
        self.record_button.config(state="normal")
    
    def toggle_recording(self):
        """Alterna entre iniciar y detener grabación"""
        if not self.is_recording:
            # Iniciar grabación
            success, msg = self.recorder.start_recording()
            
            if not success:
                messagebox.showerror("Error", msg)
                return
            
            self.is_recording = True
            self.record_button.config(text="⏹ Detener Grabación")
            self.status_label.config(text="🔴 Grabando...", foreground="red")
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(1.0, "Grabando audio...\n")
            
        else:
            # Detener grabación
            self.is_recording = False
            self.record_button.config(text="🎤 Iniciar Grabación", state="disabled")
            self.status_label.config(text="⏳ Procesando transcripción...", foreground="orange")
            
            # Procesar en segundo plano
            threading.Thread(target=self.process_audio, daemon=True).start()
    
    def process_audio(self):
        """Procesa el audio y realiza la transcripción"""
        try:
            # Detener y guardar grabación
            audio_file, msg = self.recorder.stop_recording()
            
            if audio_file is None:
                self.root.after(0, lambda: self.show_error(msg))
                return
            
            # Actualizar UI
            self.root.after(0, lambda: self.text_box.insert(
                tk.END, f"\n{msg}\nTranscribiendo...\n"
            ))
            
            # Transcribir
            transcription, msg = self.recorder.transcribe(audio_file)
            
            if transcription is None:
                self.root.after(0, lambda: self.show_error(msg))
                return
            
            # Actualizar con transcripción
            self.root.after(0, lambda: self.update_transcription(transcription))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error inesperado: {str(e)}"))
    
    def update_transcription(self, text):
        """Actualiza el área de texto con la transcripción"""
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(1.0, text)
        self.status_label.config(text="✓ Transcripción completada", foreground="green")
        self.record_button.config(state="normal")
    
    def show_error(self, error):
        """Muestra un error en la interfaz"""
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(1.0, f"❌ Error: {error}")
        self.status_label.config(text="Error", foreground="red")
        self.record_button.config(state="normal")
        messagebox.showerror("Error", error)
    
    def on_closing(self):
        """Limpia recursos al cerrar la aplicación"""
        self.recorder.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
