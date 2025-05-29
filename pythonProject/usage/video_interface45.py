import tkinter as tk
import customtkinter
import os
import subprocess
import platform
import cv2
import threading
import pandas as pd
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from emotion_video_track import EmotionVideoTrack
from reform_db import reform
from datetime import date
import json
import datetime  # Импортируем модуль datetime

you_password = "123456"  # Default password
settings_password = "654321"  # Default settings password
SETTINGS_FILE = "settings.json"  # Define the settings file


class PasswordAuth:
    def __init__(self):
        self.root = customtkinter.CTk()
        customtkinter.set_appearance_mode("light")
        customtkinter.set_default_color_theme("red.json")
        self.root.title("Авторизация")
        self.root.attributes('-fullscreen', True)  # Открываем в полноэкранном режиме
        self.root.bind("<Escape>", self.exit_fullscreen)  # Выход из полноэкранного режима по нажатию Escape
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")

        self.password_label = customtkinter.CTkLabel(master=self.root, text="Введите ключ доступа:")
        self.password_label.pack(pady=10)

        self.password_entry = customtkinter.CTkEntry(master=self.root, show="*", width=200)
        self.password_entry.pack(pady=10)

        self.submit_button = customtkinter.CTkButton(master=self.root, text="Ввод", command=self.check_password)
        self.submit_button.pack(pady=10)

        self.close_button = customtkinter.CTkButton(master=self.root, text="Закрыть приложение",
                                                    command=self.close_application)
        self.close_button.pack(pady=10)
        self.load_settings()

    def load_settings(self):
        """Loads settings, including the passwords, from the settings file."""
        global you_password, settings_password
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                you_password = settings.get("password", you_password)  # Load the main password
                settings_password = settings.get("settings_password", settings_password)  # Load the settings password
        except FileNotFoundError:
            # If the file doesn't exist, use default passwords and create the file
            self.save_settings()
        except json.JSONDecodeError:
            messagebox.showerror("Ошибка", "Ошибка при чтении файла настроек. Использован пароль по умолчанию.")

    def save_settings(self):
        """Saves the passwords to the settings file."""
        settings = {"password": you_password, "settings_password": settings_password}
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении пароля в файл настроек: {e}")

    def check_password(self):
        password = self.password_entry.get()
        if password == you_password:
            self.root.destroy()
            EmotionTrackerGUI(self.screen_width, self.screen_height).root.mainloop()
        else:
            messagebox.showerror("Ошибка", "Неверный ключ")

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)

    def close_application(self):
        self.root.destroy()


class EmotionTrackerGUI:
    def __init__(self, screen_width, screen_height):
        self.loaded_output_folder = None
        customtkinter.set_appearance_mode("light")
        customtkinter.set_default_color_theme("red.json")

        self.root = customtkinter.CTk()
        self.root.title("MARCIPAN")
        self.root.attributes('-fullscreen', True)  # Открываем в полноэкранном режиме
        self.root.bind("<Escape>", self.exit_fullscreen)  # Выход из полноэкранного режима по нажатию Escape
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")

        self.main_frame = None  # Initialize main_frame to None
        self.results_frame = None  # Initialize results_frame to None
        self.settings_frame = None  # Initialize settings frame
        self.emotion_tracker = EmotionVideoTrack()

        self.is_recording = False
        self.camera_thread = None
        self.cap = None  # VideoCapture object
        self.temp_video_path = "temp_camera_video.mp4"
        self.processed_video_path = "temp_processed_video.mp4"  # temporary file for processed video

        # Load settings from file
        self.load_settings()

        self.annotations_file = customtkinter.StringVar(value="face_detections.csv")
        self.output_video = customtkinter.StringVar(value="emotion_prediction_viz.mp4")
        self.status_var = customtkinter.StringVar(value="Приложение готово к работе")
        self.start_record_button = None
        self.process_button = None
        self.camera_label = None  # Label для отображения видео с камеры
        self.progress_bar = None  # Шкала прогресса
        self.progress_label = None  # Label for progress bar
        self.buttons_frame = None  # Store the buttons_frame

        # Date entries' values will be stored in these variables
        self.year_str = tk.StringVar()
        self.month_str = tk.StringVar()
        self.day_str = tk.StringVar()

        self.year_entry = None
        self.month_entry = None
        self.day_entry = None

        # Person info entry's values
        self.fio_entry = None
        self.gender_entry = None
        self.birthdate_entry = None

        self.current_output_folder = None  # Store the dynamically generated output folder path

        self.logo_image = self.load_logo()  # Load the logo image
        self.logo_label = None  # Initialize the logo label

        self.create_main_frame()

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)

    def load_logo(self):
        """Loads and resizes the logo image."""
        try:
            image = Image.open("1603630.png")  # Replace with your logo file name
            # Resize image
            width, height = image.size
            new_width = int(width * 1.2)  # Adjust the scale factor as needed
            new_height = int(height * 1.2)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            return ImageTk.PhotoImage(image)
        except FileNotFoundError:
            messagebox.showerror("Ошибка", "Не удалось загрузить логотип.")
            return None  # Return None if the image fails to load

    def create_logo(self):
        """Creates the logo label on the root window and places it in the top-left corner."""
        if self.logo_image:
            self.logo_label = customtkinter.CTkLabel(self.root, image=self.logo_image, text="")  # Ensure no default text
            self.logo_label.place(x=80, y=30, anchor="nw")
            self.logo_label.lift()  # Ensure the logo is always on top

    def remove_logo(self):
        """Removes the logo label."""
        if self.logo_label:
            self.logo_label.destroy()
            self.logo_label = None

    def lock_application(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()  # Освобождаем камеру
        cv2.destroyAllWindows()
        self.root.destroy()
        PasswordAuth().root.mainloop()

    def browse_video(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
            self.show_process_button()

    def clear_video_path(self):
        """Очищает путь к видеофайлу и скрывает кнопку обработки."""
        self.video_path.set("")
        self.hide_process_button()

    def load_settings(self):
        """Loads settings from the settings file."""
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                self.output_folder = customtkinter.StringVar(
                    value=settings.get("output_folder", "база данных результатов"))
                global you_password, settings_password
                you_password = settings.get("password", you_password)
                settings_password = settings.get("settings_password", settings_password)
        except FileNotFoundError:
            # If the file doesn't exist, create a default settings file
            self.output_folder = customtkinter.StringVar(value="база данных результатов")
            self.save_settings()  # Create the settings file
        except json.JSONDecodeError:
            messagebox.showerror("Ошибка", "Ошибка при чтении файла настроек. Использованы настройки по умолчанию.")
            self.output_folder = customtkinter.StringVar(value="база данных результатов")

    def save_settings(self):
        """Saves settings to the settings file."""
        settings = {
            "output_folder": self.output_folder.get(),
            "password": you_password,
            "settings_password": settings_password
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении настроек: {e}")

    def show_settings_password_dialog(self):
        """Shows a password entry dialog for accessing settings."""
        # Remove logo before showing password dialog
        self.remove_logo()

        dialog = customtkinter.CTkToplevel(self.root)
        dialog.title("Требуется авторизация")
        dialog.attributes('-fullscreen', True)  # Make dialog fullscreen

        password_label = customtkinter.CTkLabel(dialog, text="Введите ключ доступа:")
        password_label.pack(pady=5)

        password_entry = customtkinter.CTkEntry(dialog, show="*", width=200)
        password_entry.pack(pady=5)

        def check_settings_password():
            password = password_entry.get()
            global settings_password
            if password == settings_password:
                dialog.destroy()
                # Re-create the logo after closing dialog
                self.create_settings_frame()
            else:
                messagebox.showerror("Ошибка", "Неверный ключ")
                #Re-create logo even if password is wrong
                self.create_logo()

        submit_button = customtkinter.CTkButton(dialog, text="Ввод", command=check_settings_password)
        submit_button.pack(pady=5)

        # Make the dialog modal (block input to other windows)
        dialog.grab_set()
        self.root.wait_window(dialog)

        # If password incorrect or dialog canceled (unlikely due to grab_set but good practice):
        self.create_logo() # Try creating logo here too after the dialog.

    def create_settings_frame(self):
        """Creates the settings frame."""
        if self.main_frame:
            self.main_frame.destroy()
            self.remove_logo()
        elif self.results_frame:
            self.results_frame.destroy()
            self.remove_logo()

        self.settings_frame = customtkinter.CTkFrame(master=self.root)
        self.settings_frame.pack(fill="both", expand=True)

        # Recreate logo for this frame
        self.create_logo()

        customtkinter.CTkLabel(master=self.settings_frame, text="Настройки", font=('Roboto', 35)).pack(pady=20)

        # Change main password
        customtkinter.CTkLabel(master=self.settings_frame, text="Новый пароль для входа:").pack(pady=5)
        new_password_entry = customtkinter.CTkEntry(master=self.settings_frame, show="*", width=200)
        new_password_entry.pack(pady=5)

        # Change settings password
        customtkinter.CTkLabel(master=self.settings_frame, text="Новый пароль для настроек:").pack(pady=5)
        new_settings_password_entry = customtkinter.CTkEntry(master=self.settings_frame, show="*", width=200)
        new_settings_password_entry.pack(pady=5)

        def change_passwords():
            global you_password, settings_password
            new_password = new_password_entry.get()
            new_settings_password = new_settings_password_entry.get()

            if new_password:
                you_password = new_password
            if new_settings_password:
                settings_password = new_settings_password

            self.save_settings()
            messagebox.showinfo("Успешно", "Пароль успешно изменен.")

        customtkinter.CTkButton(master=self.settings_frame, text="Изменить пароли", command=change_passwords).pack(
            pady=10)

        # Change output folder
        customtkinter.CTkLabel(master=self.settings_frame, text="Путь к папке 'база данных результатов':").pack(pady=5)
        output_folder_entry = customtkinter.CTkEntry(master=self.settings_frame, width=400)
        output_folder_entry.insert(0, self.output_folder.get())  # Display current value
        output_folder_entry.pack(pady=5)

        def change_output_folder():
            new_output_folder = output_folder_entry.get()
            if new_output_folder:
                self.output_folder.set(new_output_folder)
                self.save_settings()
                messagebox.showinfo("Успешно", "Путь к папке успешно изменен.")
            else:
                messagebox.showerror("Ошибка", "Пожалуйста, введите путь к папке.")

        customtkinter.CTkButton(master=self.settings_frame, text="Изменить путь к папке",
                                command=change_output_folder).pack(pady=10)

        # Back button
        customtkinter.CTkButton(master=self.settings_frame, text="Назад", command=self.return_to_main_frame).pack(
            pady=20)

    def create_main_frame(self):
        """Создает главный фрейм с элементами управления."""
        if self.results_frame:
            self.results_frame.destroy()  # Destroy the results frame if it exists
            self.remove_logo()
        if self.settings_frame:
            self.settings_frame.destroy()
            self.remove_logo()

        self.main_frame = customtkinter.CTkFrame(master=self.root)
        self.main_frame.pack(pady=20, padx=60, fill="both", expand=True)

        #Create logo after destroying the frames, so it remains above
        self.create_logo()

        # Кнопка "Заблокировать" в правом верхнем углу
        self.lock_button = customtkinter.CTkButton(master=self.main_frame, text="Заблокировать",
                                                   command=self.lock_application, width=100)
        self.lock_button.pack(anchor="ne", padx=10, pady=10)

        # Settings Button
        self.settings_button = customtkinter.CTkButton(master=self.main_frame, text="Настройки",
                                                       command=self.show_settings_password_dialog, width=100)
        self.settings_button.pack(anchor="ne", padx=10, pady=10)  # Position in top left

        customtkinter.CTkLabel(
            master=self.main_frame,
            text="ДЕТЕКТОР СОСТОЯНИЯ ЧЕЛОВЕКА MARCIPAN",
            font=('Roboto', 40),
            wraplength=1000
        ).pack(pady=20)

        source_frame = customtkinter.CTkFrame(master=self.main_frame)
        source_frame.pack(fill="x", padx=20, pady=10)

        customtkinter.CTkLabel(
            source_frame,
            text="Путь к видео-файла:",
            font=('Roboto', 25)
        ).pack(pady=5)

        # Store the buttons_frame in self
        self.buttons_frame = customtkinter.CTkFrame(source_frame)
        self.buttons_frame.pack(fill="x", pady=5)

        customtkinter.CTkButton(
            self.buttons_frame,
            text="Из файла",
            command=self.show_file_input,
            width=150,
            height=55
        ).pack(side="left", padx=10)

        customtkinter.CTkButton(
            self.buttons_frame,
            text="С камеры",
            command=self.open_camera,  # Изменено
            width=150,
            height=55
        ).pack(side="left", padx=10)

        # "Прошлые результаты" button
        self.past_results_button = customtkinter.CTkButton(
            self.buttons_frame,
            text="Прошлые результаты",
            command=self.show_results,  # Use the show_results function to return to results
            width=150,
            height=55
        )
        self.past_results_button.pack(side="left", padx=10)

        self.file_frame = customtkinter.CTkFrame(master=self.main_frame)
        self.file_frame.pack(fill="x", padx=20, pady=10)

        self.video_path = customtkinter.StringVar()
        customtkinter.CTkEntry(
            self.file_frame,
            textvariable=self.video_path,
            width=400,
            placeholder_text="Путь к видео-файлу"
        ).pack(side="left", padx=10)

        customtkinter.CTkButton(
            self.file_frame,
            text="Поиск",
            command=self.browse_video,
            width=100
        ).pack(side="left", padx=10)

        self.file_frame.pack_forget()

        # Output folder input with year, month, and day
        output_frame = customtkinter.CTkFrame(master=self.main_frame)
        output_frame.pack(fill="x", padx=20, pady=10)

        # Date Input Fields
        customtkinter.CTkLabel(output_frame, text="Введите текущий год:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.year_entry = customtkinter.CTkEntry(output_frame, width=100, textvariable=self.year_str)
        self.year_entry.grid(row=0, column=1, padx=5, pady=5)

        customtkinter.CTkLabel(output_frame, text="Введите текущий месяц:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.month_entry = customtkinter.CTkEntry(output_frame, width=100, textvariable=self.month_str)
        self.month_entry.grid(row=1, column=1, padx=5, pady=5)

        customtkinter.CTkLabel(output_frame, text="Введите текущий день:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.day_entry = customtkinter.CTkEntry(output_frame, width=100, textvariable=self.day_str)
        self.day_entry.grid(row=2, column=1, padx=5, pady=5)

        # Person Info Input Fields, placed to the right of the date input
        customtkinter.CTkLabel(output_frame, text="ФИО:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.fio_entry = customtkinter.CTkEntry(output_frame, width=200)
        self.fio_entry.grid(row=0, column=3, padx=5, pady=5)

        customtkinter.CTkLabel(output_frame, text="Пол:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.gender_entry = customtkinter.CTkEntry(output_frame, width=200)
        self.gender_entry.grid(row=1, column=3, padx=5, pady=5)

        customtkinter.CTkLabel(output_frame, text="Дата рождения:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.birthdate_entry = customtkinter.CTkEntry(output_frame, width=200)
        self.birthdate_entry.grid(row=2, column=3, padx=5, pady=5)

        self.process_button = customtkinter.CTkButton(
            self.main_frame,
            text="Обработать видео",
            command=self.start_processing,
            width=200,
            height=55,
            font=('Roboto', 25)
        )
        self.process_button.pack(pady=10)
        self.hide_process_button()  # Hide button initially

        # Label для progress bar
        self.progress_label = customtkinter.CTkLabel(
            master=self.main_frame,
            text="Обработка видео",
            font=('Roboto', 35)
        )
        self.progress_label.pack(pady=(10, 0))  # Add some padding on top, remove bottom padding
        self.progress_label.pack_forget()  # Initially hide it

        # Шкала прогресса
        self.progress_bar = customtkinter.CTkProgressBar(master=self.main_frame, width=400)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)  # Изначально прогресс равен 0
        self.progress_bar.pack_forget()  # Hide progress bar initially

        self.status_label = customtkinter.CTkLabel(
            self.main_frame,
            textvariable=self.status_var,
            font=('Roboto', 25)
        )
        self.status_label.pack(pady=10)

        self.recording_status = customtkinter.StringVar(value="")
        self.recording_label = customtkinter.CTkLabel(
            self.main_frame,
            textvariable=self.recording_status,
            font=('Roboto', 25),
            text_color="red"
        )
        self.recording_label.pack(pady=5)

        # Label для отображения видеопотока с камеры
        self.camera_label = tk.Label(self.main_frame)
        self.camera_label.pack(side=tk.BOTTOM, anchor="se", padx=10, pady=10)
        self.camera_label.place(relx=1.0, rely=1.0, anchor='se')  # Position in bottom right

    def show_process_button(self):
        self.process_button.pack(pady=10)

    def hide_process_button(self):
        self.process_button.pack_forget()

    def show_file_input(self):
        self.clear_video_path()  # Clear the video path when selecting "Из файла"
        self.file_frame.pack(fill="x", padx=20, pady=10)
        if self.is_recording:
            self.stop_camera_recording()

    def open_camera(self):
        self.clear_video_path()  # Clear video path when opening camera
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  # Открываем камеру

            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть камеру")
                return

            self.update_camera_feed()  # Начинаем обновление видео

            self.start_record_button = customtkinter.CTkButton(self.main_frame, text="Начать запись",
                                                               command=self.start_camera_recording, width=150)
            self.start_record_button.pack(pady=10)

    def close_camera(self):
        """Останавливает захват видео и очищает camera_label."""
        if self.is_recording:
            self.stop_camera_recording()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.camera_label:
            self.camera_label.destroy()
            self.camera_label = None

    def update_camera_feed(self):
        """Обновляет видеопоток с камеры в camera_label."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Ошибка", "Не удалось получить кадр с камеры")
                self.close_camera()
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)  # Используем config вместо configure
            self.camera_label.after(10, self.update_camera_feed)  # Обновляем каждые 10 мс
        else:
            self.close_camera()

    def start_camera_recording(self):
        self.is_recording = True
        self.start_record_button.configure(text="Остановить запись", command=self.stop_camera_recording)
        self.file_frame.pack_forget()
        self.recording_status.set("Запись с камеры...")
        self.camera_thread = threading.Thread(target=self.record_from_camera)
        self.camera_thread.start()

    def stop_camera_recording(self):
        self.is_recording = False
        self.recording_status.set("Запись остановлена. Видео сохранено.")
        self.start_record_button.destroy()
        self.show_process_button()

    def record_from_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  # Open the camera here if it's not already open
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Attempt to get FPS, handle potential error
        try:
            fps = int(self.cap.get(cv2.CAP_PROP_FRAME_FPS))
        except AttributeError:
            fps = 30  # Set a default FPS if it can't be read

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))

        while self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                out.write(frame)
            else:
                break

        out.release()
        self.video_path.set(self.temp_video_path)
        self.cap.release()  # Release the camera here
        cv2.destroyAllWindows()  # close windows

    def start_processing(self):
        """Запускает обработку видео в отдельном потоке."""
        if not self.video_path.get():
            messagebox.showerror("Ошибка", "Пожалуйста, выберите видео-файл")
            return

        # Get the date from the input fields
        year_str = self.year_entry.get()
        month_str = self.month_entry.get()
        day_str = self.day_entry.get()

        # Validate date input (basic validation and range check)
        if not (year_str and month_str and day_str) or not (
                year_str.isdigit() and month_str.isdigit() and day_str.isdigit()):
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректную дату (год, месяц, день) цифрами.")
            return

        year = int(year_str)
        month = int(month_str)
        day = int(day_str)

        try:
            date(year, month, day)  # This will raise ValueError if the date is invalid
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Введена некорректная дата: {e}")
            return

        # Get FIO from input field
        fio = self.fio_entry.get()

        if not fio:
            messagebox.showerror("Ошибка", "Пожалуйста, введите ФИО.")
            return

        # Get current time (HH.MM)
        now = datetime.datetime.now()
        current_time = now.strftime("%H.%M")  # Format as HH.MM

        # Construct the output folder path including FIO and time
        base_output_folder = self.output_folder.get()  # Get the base folder
        output_folder = os.path.join(base_output_folder, str(year), str(month).zfill(2), str(day).zfill(2),
                                     f"{fio} {current_time}")  # Add time to folder name
        self.current_output_folder = output_folder

        # **ОБЯЗАТЕЛЬНО: Создаём директорию ПЕРЕД сохранением person_info.txt**
        os.makedirs(output_folder, exist_ok=True)

        person_info = {
            "ФИО": self.fio_entry.get(),
            "Дата рождения": self.birthdate_entry.get(),
            "Пол": self.gender_entry.get()
        }

        person_info_path = os.path.join(output_folder, 'person_info.txt')
        try:
            with open(person_info_path, 'w') as f:
                for key, value in person_info.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении информации о человеке: {e}")

        # Hide main frame widgets
        for widget in self.main_frame.winfo_children():
            widget.pack_forget()

        # Repack the lock button
        self.lock_button.pack(anchor="ne", padx=10, pady=10)

        self.progress_label.pack(pady=(10, 0))  # Show and position the label
        self.progress_bar.pack(pady=5)  # Show and position the progress bar
        self.progress_bar.set(0)  # Reset progress

        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        """Обрабатывает видео."""
        video_path = self.video_path.get()
        output_folder = self.current_output_folder  # Use the stored output folder path, because self.output_folder is customtkinterStringVar
        annotations_file = self.annotations_file.get()
        output_video = self.output_video.get()

        os.makedirs(output_folder, exist_ok=True)

        try:
            self.status_var.set("Обработка...")
            self.root.update()

            def progress_callback(frame_num, total_frames):
                progress = (frame_num + 1) / total_frames
                self.root.after(0, self.progress_bar.set, progress)
                self.root.after(0, self.status_var.set, f"Обработка: {int(progress * 100)}%")

            output_folder = self.current_output_folder
            self.emotion_tracker.track_emotions_on_video(
                video_path,
                output_folder,
                annotations_file,
                output_video,
                progress_callback=progress_callback  # Pass the callback function
            )
            self.convert_csv_to_xls(output_folder=output_folder)

            self.status_var.set("Обработка была завершена!\nПриложение готово к работе")
            messagebox.showinfo("Успешно", "Обработка видео завершена успешно!")

            # Destroy progress bar and label after processing
            self.progress_bar.destroy()
            self.progress_label.destroy()

            self.show_results()

        except Exception as e:
            self.status_var.set("Обнаружена ошибка")
            messagebox.showerror("Ошибка", f"Обнаружена ошибка: {str(e)}")
            # Destroy progress bar and label on error
            if self.progress_bar:
                self.progress_bar.destroy()
            if self.progress_label:
                self.progress_label.destroy()

    def convert_csv_to_xls(self, output_folder="output_preds"):
        csv_file_path = os.path.join(output_folder, self.annotations_file.get())
        xlsx_file_path = os.path.join(output_folder, self.annotations_file.get().replace('.csv', '.xlsx'))

        df = pd.read_csv(csv_file_path)
        df.to_excel(xlsx_file_path, index=False)
        reform(output_folder=output_folder)

    def show_results(self):
        """Displays the results in a new frame on top of the main frame."""
        if self.main_frame:
            self.main_frame.destroy()  # Destroy the main frame

        self.results_frame = customtkinter.CTkFrame(master=self.root)
        self.results_frame.pack(pady=20, padx=60, fill="both", expand=True)

        customtkinter.CTkLabel(
            master=self.results_frame,
            text="Сгенерированные файлы:",
            font=('Roboto', 25)
        ).pack(pady=10)

        annotations_path = os.path.join(self.current_output_folder, self.annotations_file.get())
        xlsx_path = os.path.join(self.current_output_folder, self.annotations_file.get().replace('.csv', '.xlsx'))
        video_path = os.path.join(self.current_output_folder, self.output_video.get())
        persons_faces_path = os.path.join(self.current_output_folder, "persons_faces")

        for label, path in [
            ("Полный анализ:", annotations_path),
            ("Краткий анализ с рекомендациями:", xlsx_path),
            ("Видео с результатом:", video_path)
        ]:
            file_frame = customtkinter.CTkFrame(self.results_frame)
            file_frame.pack(fill="x", pady=5)

            customtkinter.CTkLabel(
                file_frame,
                text=label,
                font=('Roboto', 25, 'bold')
            ).pack(side="left", padx=5)

            customtkinter.CTkLabel(
                file_frame,
                text=path,
                font=('Roboto', 25)
            ).pack(side="left", padx=5)

            customtkinter.CTkButton(
                file_frame,
                text="Открыть",
                command=lambda p=path: self.open_file(p),
                width=80
            ).pack(side="right", padx=5)

        customtkinter.CTkButton(
            self.results_frame,
            text="Открыть папку вывода",
            command=lambda: self.open_folder(self.current_output_folder),  # Use self.current_output_folder
            width=200
        ).pack(pady=20)

        persons_faces_path = os.path.join(self.current_output_folder,
                                          "persons_faces")  # Use self.current_output_folder
        if os.path.exists(persons_faces_path):
            customtkinter.CTkButton(
                self.results_frame,
                text="Обнаруженные люди",
                command=lambda: self.open_folder(persons_faces_path),  # Use persons_faces_path
                width=200
            ).pack(pady=10)

        # Add "Back" button
        self.back_button = customtkinter.CTkButton(master=self.results_frame, text="Назад",
                                                   command=self.return_to_main_frame, width=100)
        self.back_button.pack(pady=20)

        self.display_excel_data(xlsx_path, self.results_frame)

    def return_to_main_frame(self):
        """Returns to the main frame by destroying the results frame."""
        if self.results_frame:
            self.results_frame.destroy()
            self.results_frame = None  # Reset results_frame

        self.create_main_frame()

    def display_excel_data(self, file_path, frame):
        df = pd.read_excel(file_path)

        text_box = customtkinter.CTkTextbox(frame, width=700, height=300, font=('Roboto', 25))
        text_box.pack(pady=20)

        headers = df.columns.tolist()
        header_string = " | ".join(headers)
        text_box.insert("end", header_string + "\n")
        text_box.insert("end", "-" * len(header_string) + "\n")

        for index, row in df.iterrows():
            row_string = " | ".join([str(value) for value in row])
            text_box.insert("end", row_string + "\n")

        text_box.configure(state="disabled")

    def open_file(self, path):
        if platform.system() == 'Darwin':
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':
            os.startfile(path)
        else:
            subprocess.call(('xdg-open', path))

    def open_folder(self, path):
        if platform.system() == 'Darwin':
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':
            subprocess.Popen(f'explorer "{path}"')
        else:
            subprocess.call(('xdg-open', path))
            # Load settings from file
            self.load_settings()

            # Initialize StringVar with loaded value
            self.output_folder = customtkinter.StringVar(value=self.loaded_output_folder)
            # Initialize StringVar for date Entrys
            self.year_str = tk.StringVar()
            self.month_str = tk.StringVar()
            self.day_str = tk.StringVar()

            self.annotations_file = customtkinter.StringVar(value="face_detections.csv")
            self.output_video = customtkinter.StringVar(value="emotion_prediction_viz.mp4")
            self.status_var = customtkinter.StringVar(value="Приложение готово к работе")
            self.start_record_button = None
            self.process_button = None
            self.camera_label = None  # Label для отображения видео с камеры
            self.progress_bar = None  # Шкала прогресса
            self.progress_label = None  # Label for progress bar
            self.buttons_frame = None  # Store the buttons_frame

            self.year_entry = None
            self.month_entry = None
            self.day_entry = None
            self.current_output_folder = None  # Store the dynamically generated output folder path

            self.create_main_frame()

def main():
    PasswordAuth().root.mainloop()

if __name__ == "__main__":
    main()
