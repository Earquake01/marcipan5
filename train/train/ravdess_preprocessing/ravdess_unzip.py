import glob
import os
import zipfile


def ravdess_unzip(ravdess_path, extract_folder):
    """Разархивирует dataset RAVDESS в указанный каталог.

    Parameters
    ----------
    ravdess_path : str
        путь к папке содержащей dataset RAVDESS
    extract_folder : str
        путь к папке куда извлекаем данные из dataset'а RAVDESS
    """
    mask_path = os.path.join(ravdess_path, f"*.zip")
    zip_files = glob.glob(mask_path)

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_folder)


if __name__ == "__main__":
    # разархивируем весь датасет
    ravdess_path = "RAVDESS_src"
    extract_folder = "RAVDESS"

    ravdess_unzip(ravdess_path, extract_folder)
