
import tkinter
import customtkinter
from emotion_video_track import EmotionVideoTrack
import os
from tkinter import filedialog, messagebox
import subprocess
import platform
import cv2
import threading
import pandas as pd  # Импортируем pandas для работы с данными
from reform_db import reform
you_password = "123456"

class PasswordAuth:
    def __init__(self):
        self.root = customtkinter.CTk()
        self.root.title("Авторизация")
        self.root.geometry("1280x720")

        self.password_label = customtkinter.CTkLabel(master=self.root, text="Введите ключ доступа:")
        self.password_label.pack(pady=10)

        self.password_entry = customtkinter.CTkEntry(master=self.root, show="*", width=200)
        self.password_entry.pack(pady=10)

        self.submit_button = customtkinter.CTkButton(master=self.root, text="Ввод", command=self.check_password)
        self.submit_button.pack(pady=10)

    def check_password(self):
        password = self.password_entry.get()
        if password == you_password:  # replace with your desired password
            self.root.destroy()
            EmotionTrackerGUI().root.mainloop()
        else:
            messagebox.showerror("Ошибка", "Неверный ключ")

class EmotionTrackerGUI:
    def __init__(self):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.root = customtkinter.CTk()
        self.root.geometry("1280x720")
        self.root.title("MARCIPAN")

        self.main_frame = customtkinter.CTkFrame(master=self.root)
        self.main_frame.pack(pady=20, padx=60, fill="both", expand=True)

        self.emotion_tracker = EmotionVideoTrack()

        self.is_recording = False
        self.camera_thread = None
        self.temp_video_path = "temp_camera_video.mp4"

        self.output_folder = customtkinter.StringVar(value="output_preds")
        self.annotations_file = customtkinter.StringVar(value="face_detections.csv")
        self.output_video = customtkinter.StringVar(value="emotion_prediction_viz.mp4")
        self.status_var = customtkinter.StringVar(value="Готово")

        self.create_widgets()

    def browse_video(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)

    def create_widgets(self):
        customtkinter.CTkLabel(
            master=self.main_frame,
            text="ДЕТЕКТОР СОСТОЯНИЯ ЧЕЛОВЕКА MARCIPAN",
            font=('Arial', 30),
            wraplength=1000
        ).pack(pady=20)

        source_frame = customtkinter.CTkFrame(master=self.main_frame)
        source_frame.pack(fill="x", padx=20, pady=10)

        customtkinter.CTkLabel(
            source_frame,
            text="Путь к видео-файла:",
            font=('Arial', 14)
        ).pack(pady=5)

        buttons_frame = customtkinter.CTkFrame(source_frame)
        buttons_frame.pack(fill="x", pady=5)

        customtkinter.CTkButton(
            buttons_frame,
            text="Из файла",
            command=self.show_file_input,
            width=150
        ).pack(side="left", padx=10)

        customtkinter.CTkButton(
            buttons_frame,
            text="С камеры",
            command=self.toggle_camera_recording,
            width=150
        ).pack(side="left", padx=10)

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

        output_frame = customtkinter.CTkFrame(master=self.main_frame)
        output_frame.pack(fill="x", padx=20, pady=10)

        customtkinter.CTkLabel(
            output_frame,
            text="Вывод-папка:"
        ).pack(pady=5)

        customtkinter.CTkEntry(
            output_frame,
            textvariable=self.output_folder,
            width=400,
            placeholder_text="Папка с результатами"
        ).pack()

        customtkinter.CTkLabel(
            output_frame,
            text="Файл-аннотация:"
        ).pack(pady=5)

        customtkinter.CTkEntry(
            output_frame,
            textvariable=self.annotations_file,
            width=400,
            placeholder_text="Имя файла-аннотации"
        ).pack()

        customtkinter.CTkLabel(
            output_frame,
            text="Видео с результатом:"
        ).pack(pady=5)

        customtkinter.CTkEntry(
            output_frame,
            textvariable=self.output_video,
            width=400,
            placeholder_text="Имя видео с результатом"
        ).pack()

        customtkinter.CTkButton(
            self.main_frame,
            text="Обработать видео",
            command=self.process_video,
            width=200,
            height=40,
            font=('Arial', 15)
        ).pack(pady=20)

        self.status_label = customtkinter.CTkLabel(
            self.main_frame,
            textvariable=self.status_var,
            font=('Arial', 15)
        )
        self.status_label.pack(pady=10)

        self.recording_status = customtkinter.StringVar(value="")
        self.recording_label = customtkinter.CTkLabel(
            self.main_frame,
            textvariable=self.recording_status,
            font=('Arial', 15),
            text_color="red"
        )
        self.recording_label.pack(pady=5)

    def show_file_input(self):
        self.file_frame.pack(fill="x", padx=20, pady=10)
        if self.is_recording:
            self.stop_camera_recording()

    def toggle_camera_recording(self):
        if not self.is_recording:
            self.start_camera_recording()
        else:
            self.stop_camera_recording()

    def start_camera_recording(self):
        self.is_recording = True
        self.file_frame.pack_forget()
        self.recording_status.set("Записываем с камеры... Нажмите еще раз кнопку 'Из камеры', чтобы завершить запись")
        self.camera_thread = threading.Thread(target=self.record_from_camera)
        self.camera_thread.start()

    def stop_camera_recording(self):
        self.is_recording = False
        self.recording_status.set("Запись остановлена. Обрабатываем видео...")
        self.video_path.set(self.temp_video_path)

    def record_from_camera(self):
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))

        while self.is_recording:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def process_video(self):
        if not self.video_path.get():
            messagebox.showerror("Ошибка", "Пожалуйста, выберите видео-файл")
            return

        os.makedirs(self.output_folder.get(), exist_ok=True)

        try:
            self.status_var.set("Обработка...")
            self.root.update()

            self.emotion_tracker.track_emotions_on_video(
                self.video_path.get(),
                self.output_folder.get(),
                self.annotations_file.get(),
                self.output_video.get()
            )

            self.convert_csv_to_xls()  # Конвертируем CSV в XLSX

            self.status_var.set("Обработка завершена!")
            messagebox.showinfo("Успешно", "Обработка видео завершена успешно!")

            self.show_results()

        except Exception as e:
            self.status_var.set("Обнаружена ошибка")
            messagebox.showerror("Ошибка", f"Обнаружена ошибка: {str(e)}")

    def convert_csv_to_xls(self):
        csv_file_path = os.path.join(self.output_folder.get(), self.annotations_file.get())
        xlsx_file_path = os.path.join(self.output_folder.get(), self.annotations_file.get().replace('.csv', '.xlsx'))

        # Читаем CSV файл
        df = pd.read_csv(csv_file_path)

        # Сохраняем в формате XLSX
        df.to_excel(xlsx_file_path, index=False)
        reform()

    def show_results(self):
        results_window = customtkinter.CTkToplevel(self.root)
        results_window.geometry("800x500")
        results_window.title("Результаты обработки")

        results_frame = customtkinter.CTkFrame(master=results_window)
        results_frame.pack(pady=20, padx=20, fill="both", expand=True)

        customtkinter.CTkLabel(
            results_frame,
            text="Сгенерированные файлы:",
            font=('Arial', 20)
        ).pack(pady=10)

        # Обработка выводимых файлов
        annotations_path = os.path.join(self.output_folder.get(), self.annotations_file.get())
        xlsx_path = os.path.join(self.output_folder.get(), self.annotations_file.get().replace('.csv', '.xlsx'))
        video_path = os.path.join(self.output_folder.get(), self.output_video.get())

        for label, path in [
            ("Полный анализ:", annotations_path),
            ("Краткий анализ с рекомендациями:", xlsx_path),
            ("Видео с результатом:", video_path)
        ]:
            file_frame = customtkinter.CTkFrame(results_frame)
            file_frame.pack(fill="x", pady=5)

            customtkinter.CTkLabel(
                file_frame,
                text=label,
                font=('Arial', 12, 'bold')
            ).pack(side="left", padx=5)

            customtkinter.CTkLabel(
                file_frame,
                text=path,
                font=('Arial', 12)
            ).pack(side="left", padx=5)

            customtkinter.CTkButton(
                file_frame,
                text="Открыть",
                command=lambda p=path: self.open_file(p),
                width=80
            ).pack(side="right", padx=5)

        customtkinter.CTkButton(
            results_frame,
            text="Открыть папку вывода",
            command=lambda: self.open_folder(self.output_folder.get()),
            width=200
        ).pack(pady=20)

        # Добавляем кнопку для открытия папки persons_faces
        persons_faces_path = os.path.join(self.output_folder.get(), "persons_faces")
        if os.path.exists(persons_faces_path):  # Проверяем существование папки
            customtkinter.CTkButton(
                results_frame,
                text="Обнаруженные люди",
                command=lambda: self.open_folder(persons_faces_path),
                width=200
            ).pack(pady=10)

        # Добавляем вывод содержимого файла face_detections.xlsx
        self.display_excel_data(xlsx_path, results_frame)

    def display_excel_data(self, file_path, frame):
        # Загружаем данные из xlsx файла
        df = pd.read_excel(file_path)

        # Создаем новый текстовый виджет для вывода данных
        text_box = customtkinter.CTkTextbox(frame, width=700, height=300, font=('Arial', 12))
        text_box.pack(pady=20)

        # Заголовки колонок
        headers = df.columns.tolist()
        header_string = " | ".join(headers)
        text_box.insert("end", header_string + "\n")
        text_box.insert("end", "-" * len(header_string) + "\n")  # Разделитель

        # Заполняем текстовым виджетом данными из DataFrame
        for index, row in df.iterrows():
            row_string = " | ".join([str(value) for value in row])
            text_box.insert("end", row_string + "\n")

        text_box.configure(state="disabled")  # Запретить редактирование текстового виджета

    def open_file(self, path):
        if platform.system() == 'Darwin':
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':
            os.startfile(path)
        else:  # linux
            subprocess.call(('xdg-open', path))

    def open_folder(self, path):
        if platform.system() == 'Darwin':
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':
            subprocess.Popen(f'explorer "{path}"')
        else:  # linux
            subprocess.call(('xdg-open', path))

def main():
    PasswordAuth().root.mainloop()

if __name__ == "__main__":
    main()