import pandas as pd

def reform():
    # Читаем файл .xlsx
    file_path = 'output_preds/face_detections.xlsx'
    df = pd.read_excel(file_path)

    # Находим все колонки содержащие 'emo'
    emo_columns = [col for col in df.columns if 'emo' in col]

    # Находим все колонки содержащие 'score'
    score_columns = [col for col in df.columns if 'score' in col]

    # Находим уникальные эмоции для каждого person_id_emo
    modes = {col: df[col].mode()[0] for col in emo_columns}

    # Создаем новый DataFrame с результатами
    result_df = pd.DataFrame(modes, index=['mode']).T
    result_df.columns = ['mode']

    # Сохраняем результат в новый .xlsx файл
    output_file_path = 'output_preds/face_detections.xlsx'
    result_df.to_excel(output_file_path)

    input_file = 'output_preds/face_detections.xlsx'

    # Загружаем данные из Excel файла
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        exit()

    # Создадим словарь для перевода
    translation_dict = {
        'sad': 'Грусть: Проявите сочувствие, предложите поддержку.',
        'neutral': 'Нейтральность: Уважайте пространство, будьте готовы слушать.',
        'fearful': 'Страх: Создайте безопасную атмосферу, выслушайте опасения.',
        'happy': 'Счастье: Разделите радость, поздравьте.',
        'angry': 'Гнев: Сохраняйте спокойствие, дайте высказаться, обратить внимание.',
        'disgust': 'Отвращение: Поймите причину, уважайте мнение.',
        'calm': "Cпокойствие, рекомендаций не предпологается"
    }


    # Переименуем столбец person_1_emo, если необходимо
    df = df.rename(columns={'person_1_emo': 'person_ru'}, errors='ignore')

    # Переведем значения в столбце 'mode' и обработаем ошибки
    df['mode_ru'] = df['mode'].map(translation_dict).fillna('неизвестно')

    # Удалим старый столбец 'mode'
    df = df.drop(columns=['mode'], errors='ignore') # errors='ignore' handles case where 'mode' doesn't exist


    # Сохраняем итоговый файл
    output_file = 'output_preds/face_detections.xlsx'
    df.to_excel(output_file, index=False)

    # Выводим результат для проверки
    print(df.head())
    print(output_file)
    return