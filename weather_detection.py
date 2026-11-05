
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk
from ultralytics import YOLO
import pyttsx3
import threading
import time


# =====================================================
# SMART WEATHER CLASSIFICATION USING YOLOv8
# Classes: Cloudy and Sunny
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Trained model location
MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "classify"
    / "runs"
    / "smart_weather"
    / "weights"
    / "best.pt"
)

IMAGE_SIZE = 224


# =====================================================
# MAIN APPLICATION
# =====================================================

class SmartWeatherApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Smart Weather Classification Using YOLOv8"
        )

        self.root.geometry("1180x800")
        self.root.minsize(1000, 700)

        self.root.configure(bg="#101827")

        self.image_path = None
        self.original_image = None
        self.photo_image = None

        self.prediction = None
        self.confidence = {}

        self.processing_time = 0
        self.model = None
        self.speaking = False

        # Colors
        self.BG = "#101827"
        self.CARD = "#1e293b"
        self.ACCENT = "#38bdf8"
        self.WHITE = "#ffffff"
        self.MUTED = "#94a3b8"
        self.GREEN = "#22c55e"
        self.ORANGE = "#f59e0b"

        self.setup_styles()

        self.create_header()
        self.create_main_content()
        self.create_buttons()
        self.create_status_bar()

        self.load_model()

    # =================================================
    # STYLES
    # =================================================

    def setup_styles(self):

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "Weather.Horizontal.TProgressbar",
            troughcolor="#334155",
            background=self.ACCENT,
            bordercolor="#334155",
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
            thickness=14
        )

    # =================================================
    # LOAD YOLO MODEL
    # =================================================

    def load_model(self):

        try:

            if not MODEL_PATH.exists():

                raise FileNotFoundError(
                    f"Model file not found:\n{MODEL_PATH}"
                )

            self.status_label.config(
                text="Loading YOLOv8 model...",
                fg=self.ORANGE
            )

            self.root.update_idletasks()

            self.model = YOLO(str(MODEL_PATH))

            self.status_label.config(
                text="YOLOv8 Model Loaded Successfully",
                fg=self.GREEN
            )

        except Exception as error:

            self.model = None

            self.status_label.config(
                text="Model Loading Failed",
                fg="#ef4444"
            )

            messagebox.showerror(
                "Model Error",
                str(error)
            )

    # =================================================
    # HEADER
    # =================================================

    def create_header(self):

        header = tk.Frame(
            self.root,
            bg="#172554",
            height=105
        )

        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🌤 SMART WEATHER CLASSIFICATION",
            font=("Arial", 23, "bold"),
            bg="#172554",
            fg=self.WHITE
        ).pack(pady=(17, 5))

        tk.Label(
            header,
            text="YOLOv8 Based Cloudy and Sunny Image Detection System",
            font=("Arial", 11),
            bg="#172554",
            fg="#bae6fd"
        ).pack()

    # =================================================
    # MAIN CONTENT
    # =================================================

    def create_main_content(self):

        main = tk.Frame(
            self.root,
            bg=self.BG
        )

        main.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=18
        )

        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # IMAGE PREVIEW CARD

        image_card = tk.Frame(
            main,
            bg=self.CARD,
            highlightbackground="#334155",
            highlightthickness=1
        )

        image_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        tk.Label(
            image_card,
            text="IMAGE PREVIEW",
            font=("Arial", 15, "bold"),
            bg=self.CARD,
            fg=self.WHITE
        ).pack(pady=15)

        self.image_frame = tk.Frame(
            image_card,
            bg="#0f172a"
        )

        self.image_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )

        self.image_label = tk.Label(
            self.image_frame,
            text="🌤\n\nUpload a Weather Image\n\nSupported: JPG, JPEG, PNG",
            font=("Arial", 16),
            bg="#0f172a",
            fg=self.MUTED,
            justify="center"
        )

        self.image_label.pack(
            fill="both",
            expand=True
        )

        # RESULT CARD

        result_card = tk.Frame(
            main,
            bg=self.CARD,
            highlightbackground="#334155",
            highlightthickness=1
        )

        result_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(12, 0)
        )

        tk.Label(
            result_card,
            text="CLASSIFICATION RESULT",
            font=("Arial", 15, "bold"),
            bg=self.CARD,
            fg=self.WHITE
        ).pack(pady=(18, 25))

        self.result_label = tk.Label(
            result_card,
            text="Waiting for Image...",
            font=("Arial", 21, "bold"),
            bg=self.CARD,
            fg=self.ACCENT,
            wraplength=300
        )

        self.result_label.pack(pady=10)

        tk.Label(
            result_card,
            text="PREDICTION CONFIDENCE",
            font=("Arial", 11, "bold"),
            bg=self.CARD,
            fg=self.MUTED
        ).pack(pady=(25, 15))

        # CLOUDY CONFIDENCE

        self.cloudy_label = tk.Label(
            result_card,
            text="☁ Cloudy: 0.00%",
            font=("Arial", 12, "bold"),
            bg=self.CARD,
            fg=self.WHITE
        )

        self.cloudy_label.pack(
            anchor="w",
            padx=22,
            pady=(5, 7)
        )

        self.cloudy_bar = ttk.Progressbar(
            result_card,
            style="Weather.Horizontal.TProgressbar",
            maximum=100
        )

        self.cloudy_bar.pack(
            fill="x",
            padx=22
        )

        # SUNNY CONFIDENCE

        self.sunny_label = tk.Label(
            result_card,
            text="☀ Sunny: 0.00%",
            font=("Arial", 12, "bold"),
            bg=self.CARD,
            fg=self.WHITE
        )

        self.sunny_label.pack(
            anchor="w",
            padx=22,
            pady=(22, 7)
        )

        self.sunny_bar = ttk.Progressbar(
            result_card,
            style="Weather.Horizontal.TProgressbar",
            maximum=100
        )

        self.sunny_bar.pack(
            fill="x",
            padx=22
        )

        # PROCESSING TIME

        tk.Label(
            result_card,
            text="PROCESSING TIME",
            font=("Arial", 11, "bold"),
            bg=self.CARD,
            fg=self.MUTED
        ).pack(pady=(35, 8))

        self.time_label = tk.Label(
            result_card,
            text="0.00 seconds",
            font=("Arial", 15, "bold"),
            bg=self.CARD,
            fg=self.GREEN
        )

        self.time_label.pack()

        # VOICE STATUS

        self.voice_label = tk.Label(
            result_card,
            text="🔊 Voice Output Ready",
            font=("Arial", 11),
            bg=self.CARD,
            fg=self.ACCENT
        )

        self.voice_label.pack(pady=22)

    # =================================================
    # BUTTONS
    # =================================================

    def create_buttons(self):

        button_frame = tk.Frame(
            self.root,
            bg=self.BG
        )

        button_frame.pack(
            fill="x",
            padx=25,
            pady=(0, 15)
        )

        buttons = [

            ("📁 Upload Image", self.upload_image, "#2563eb"),

            ("🔍 Classify Image", self.classify_image, "#0891b2"),

            ("🔊 Speak Result", self.speak_result, "#16a34a"),

            ("🔄 Reset", self.reset, "#d97706"),

            ("❌ Exit", self.exit_app, "#dc2626")

        ]

        for index, (text, command, color) in enumerate(buttons):

            button_frame.grid_columnconfigure(
                index,
                weight=1
            )

            button = tk.Button(
                button_frame,
                text=text,
                command=command,
                font=("Arial", 10, "bold"),
                bg=color,
                fg=self.WHITE,
                activebackground=color,
                activeforeground=self.WHITE,
                relief="flat",
                cursor="hand2",
                padx=5,
                pady=13
            )

            button.grid(
                row=0,
                column=index,
                sticky="ew",
                padx=5
            )

    # =================================================
    # STATUS BAR
    # =================================================

    def create_status_bar(self):

        self.status_label = tk.Label(
            self.root,
            text="Initializing application...",
            font=("Arial", 10),
            bg="#172554",
            fg=self.WHITE,
            anchor="w",
            padx=15,
            pady=8
        )

        self.status_label.pack(
            fill="x",
            side="bottom"
        )

    # =================================================
    # UPLOAD IMAGE
    # =================================================

    def upload_image(self):

        file_path = filedialog.askopenfilename(
            title="Select Weather Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:

            self.image_path = file_path

            self.original_image = Image.open(
                self.image_path
            ).convert("RGB")

            self.display_image()

            self.clear_results()

            self.status_label.config(
                text="Image Uploaded Successfully",
                fg=self.GREEN
            )

        except Exception as error:

            messagebox.showerror(
                "Image Error",
                str(error)
            )

    # =================================================
    # DISPLAY IMAGE
    # =================================================

    def display_image(self):

        if self.original_image is None:
            return

        self.root.update_idletasks()

        width = max(
            self.image_frame.winfo_width() - 20,
            300
        )

        height = max(
            self.image_frame.winfo_height() - 20,
            250
        )

        preview = self.original_image.copy()

        preview.thumbnail(
            (width, height),
            Image.Resampling.LANCZOS
        )

        self.photo_image = ImageTk.PhotoImage(
            preview
        )

        self.image_label.config(
            image=self.photo_image,
            text=""
        )

    # =================================================
    # CLASSIFY IMAGE
    # =================================================

    def classify_image(self):

        if self.image_path is None:

            messagebox.showwarning(
                "Warning",
                "Please upload a weather image first."
            )

            return

        if self.model is None:

            messagebox.showerror(
                "Model Error",
                "YOLOv8 model is not loaded."
            )

            return

        try:

            self.status_label.config(
                text="Classifying image...",
                fg=self.ORANGE
            )

            self.root.update_idletasks()

            start_time = time.perf_counter()

            results = self.model.predict(
                source=self.image_path,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            result = results[0]

            # Check whether the model is a classification model

            if result.probs is None:

                raise ValueError(
                    "The loaded best.pt is an object detection model, "
                    "not an image classification model. "
                    "Please use the classification best.pt generated "
                    "using yolov8n-cls.pt."
                )

            probabilities = result.probs.data.tolist()

            names = result.names

            predicted_index = result.probs.top1

            self.prediction = names[predicted_index]

            self.confidence = {}

            for index, probability in enumerate(probabilities):

                class_name = names[index]

                self.confidence[class_name] = (
                    probability * 100
                )

            self.processing_time = (
                time.perf_counter() - start_time
            )

            self.show_results()

            self.status_label.config(
                text="Image Classified Successfully",
                fg=self.GREEN
            )

        except Exception as error:

            self.status_label.config(
                text="Classification Failed",
                fg="#ef4444"
            )

            messagebox.showerror(
                "Classification Error",
                str(error)
            )

    # =================================================
    # SHOW RESULTS
    # =================================================

    def show_results(self):

        self.result_label.config(
            text=f"🌤 {self.prediction}",
            fg=self.GREEN
        )

        # Read class confidence values

        cloudy_conf = 0
        sunny_conf = 0

        for class_name, value in self.confidence.items():

            if class_name.lower() == "cloudy":
                cloudy_conf = value

            elif class_name.lower() == "sunny":
                sunny_conf = value

        self.cloudy_label.config(
            text=f"☁ Cloudy: {cloudy_conf:.2f}%"
        )

        self.sunny_label.config(
            text=f"☀ Sunny: {sunny_conf:.2f}%"
        )

        self.cloudy_bar["value"] = cloudy_conf

        self.sunny_bar["value"] = sunny_conf

        self.time_label.config(
            text=f"{self.processing_time:.3f} seconds"
        )

        self.voice_label.config(
            text="🔊 Prediction Ready",
            fg=self.GREEN
        )

        # Automatically announce prediction

        self.speak_result()

    # =================================================
    # VOICE OUTPUT
    # =================================================

    def speak_result(self):

        if self.prediction is None:

            messagebox.showwarning(
                "Voice Output",
                "Please classify an image first."
            )

            return

        if self.speaking:
            return

        cloudy_conf = 0
        sunny_conf = 0

        for class_name, value in self.confidence.items():

            if class_name.lower() == "cloudy":
                cloudy_conf = value

            elif class_name.lower() == "sunny":
                sunny_conf = value

        speech_text = (
            f"The weather classification result is "
            f"{self.prediction}. "
            f"Cloudy confidence is {cloudy_conf:.2f} percent. "
            f"Sunny confidence is {sunny_conf:.2f} percent."
        )

        self.speaking = True

        self.voice_label.config(
            text="🔊 Speaking prediction...",
            fg=self.ORANGE
        )

        def speak():

            engine = None

            try:

                engine = pyttsx3.init()

                engine.setProperty("rate", 150)

                engine.setProperty("volume", 1.0)

                engine.say(speech_text)

                engine.runAndWait()

            except Exception as error:

                print("Voice Error:", error)

            finally:

                if engine is not None:

                    try:
                        engine.stop()
                    except Exception:
                        pass

                self.root.after(
                    0,
                    self.voice_finished
                )

        threading.Thread(
            target=speak,
            daemon=True
        ).start()

    # =================================================
    # VOICE COMPLETED
    # =================================================

    def voice_finished(self):

        self.speaking = False

        self.voice_label.config(
            text="🔊 Voice Output Completed",
            fg=self.GREEN
        )

    # =================================================
    # CLEAR RESULTS
    # =================================================

    def clear_results(self):

        self.prediction = None

        self.confidence = {}

        self.processing_time = 0

        self.result_label.config(
            text="Waiting for Image...",
            fg=self.ACCENT
        )

        self.cloudy_label.config(
            text="☁ Cloudy: 0.00%"
        )

        self.sunny_label.config(
            text="☀ Sunny: 0.00%"
        )

        self.cloudy_bar["value"] = 0

        self.sunny_bar["value"] = 0

        self.time_label.config(
            text="0.00 seconds"
        )

        self.voice_label.config(
            text="🔊 Voice Output Ready",
            fg=self.ACCENT
        )

    # =================================================
    # RESET APPLICATION
    # =================================================

    def reset(self):

        if self.speaking:

            messagebox.showinfo(
                "Voice Output",
                "Please wait until the voice announcement finishes."
            )

            return

        self.image_path = None

        self.original_image = None

        self.photo_image = None

        self.clear_results()

        self.image_label.config(
            image="",
            text="🌤\n\nUpload a Weather Image\n\nSupported: JPG, JPEG, PNG"
        )

        self.status_label.config(
            text="Application Reset Successfully",
            fg=self.GREEN
        )

    # =================================================
    # EXIT APPLICATION
    # =================================================

    def exit_app(self):

        answer = messagebox.askyesno(
            "Exit",
            "Are you sure you want to exit?"
        )

        if answer:

            self.root.destroy()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SmartWeatherApp(root)

    root.mainloop()
