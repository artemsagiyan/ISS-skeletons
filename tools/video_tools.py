import os
import subprocess


def convert_videos_to_mp4(folder_path):
    # Получаем список всех файлов в указанной папке
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        # Формируем полный путь к файлу
        file_path = os.path.join(folder_path, file_name)

        # Проверяем, является ли файл видеофайлом и не имеет ли уже расширение .mp4
        if os.path.isfile(file_path) and not file_name.lower().endswith('.mp4'):
            # Формируем новое имя файла с расширением .mp4
            output_file_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.mp4')

            # Формируем команду для преобразования видеофайла в формат mp4
            command = f'ffmpeg -i "{file_path}" -c:v copy -c:a copy "{output_file_path}"'

            # Запускаем команду с помощью subprocess
            subprocess.call(command, shell=True)

            # Дополнительно можно удалить исходный файл, раскомментировав следующую строку
            os.remove(file_path)

    print('Преобразование видеофайлов завершено!')