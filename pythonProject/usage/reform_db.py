import pandas as pd
import os

def reform(output_folder="output_preds"):  # Добавляем параметр output_folder со значением по умолчанию
    """
    Реформирует XLSX файл, анализируя преобладающие эмоции и создавая новый файл с рекомендациями.

    Parameters:
        output_folder (str): Путь к папке, содержащей XLSX файл и куда будет сохранен результат.
                             По умолчанию "output_preds".
    """

    # Формируем путь к файлу .xlsx
    file_path = os.path.join(output_folder, 'face_detections.xlsx')

    # Читаем файл .xlsx
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Находим все колонки содержащие 'emo'
    emo_columns = [col for col in df.columns if 'emo' in col]

    # Находим все колонки содержащие 'score'
    score_columns = [col for col in df.columns if 'score' in col]

    # Находим уникальные эмоции для каждого person_id_emo
    modes = {col: df[col].mode()[0] for col in emo_columns}

    # Создаем новый DataFrame с результатами
    result_df = pd.DataFrame(modes, index=['mode']).T
    result_df.columns = ['mode']

    # Формируем путь для сохранения нового .xlsx файла
    output_file_path = os.path.join(output_folder, 'face_detections.xlsx')
    result_df.to_excel(output_file_path)

    input_file = output_file_path

    # Загружаем данные из Excel файла
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    # Создадим словарь для перевода
    translation_dict = {
        'sad': 'Грусть: ОБРАТИТЬ ВНИМАНИЕ, выясните причину, проявите сочувствие, предложите поддержку, зададайте вопросы.',
        'neutral': 'Нейтральность: Уважайте пространство, рекомендаций во взаимодействии не предполагается.',
        'fearful': 'Страх: ОБРАТИТЬ ВНИМАНИЕ, выясните причину, создайте безопасную атмосферу, выслушайте опасения.',
        'happy': 'Счастье: Разделите радость, рекомендаций во взаимодействии не предполагается.',
        'angry': 'Гнев: ОБРАТИТЬ ВНИМАНИЕ, сохраняйте спокойствие, дайте высказаться, выясните причину, зададайте вопросы.',
        'disgust': 'Отвращение: выясните причину, зададайте вопросы.',
        'calm': 'Cпокойствие, рекомендаций не предполагается',
        'surprised': 'Удивление: Выясните причину, зададайте вопросы, разделите интерес.'
    }


    # Переименуем столбец person_1_emo, если необходимо
    df = df.rename(columns={'person_1_emo': 'person_ru'}, errors='ignore')

    # Переведем значения в столбце 'mode' и обработаем ошибки
    df['mode_ru'] = df['mode'].map(translation_dict).fillna('неизвестно')

    # Удалим старый столбец 'mode'
    df = df.drop(columns=['mode'], errors='ignore') # errors='ignore' handles case where 'mode' doesn't exist


    # Сохраняем итоговый файл
    output_file = output_file_path
    df.to_excel(output_file, index=False)

    # Выводим результат для проверки
    print(df.head())
    print(output_file)
    return