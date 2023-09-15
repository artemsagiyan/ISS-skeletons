import logging
import subprocess

import cv2
import yaml

import os

from tools.video_tools import convert_videos_to_mp4

# Пути к директориям
dir = '/home/student-2/skeletons/'
frame_dirs = [
    f'{dir}/frames/',
    f'{dir}/frames_skeletons/',
    f'{dir}/video_skeletons/',
    f'{dir}/frames_skeletons/fast_pose_ResNet50/',
    f'{dir}/video_skeletons/fast_pose_ResNet50/',
    f'{dir}/frames_skeletons/simple_baseline_ResNet50/',
    f'{dir}/video_skeletons/simple_baseline_ResNet50/',
    f'{dir}/frames_skeletons/fast_pose_duc_ResNet50/',
    f'{dir}/video_skeletons/fast_pose_duc_ResNet50/',
    f'{dir}/frames_skeletons/fast_pose_duc_ResNet50_unshuffle/',
    f'{dir}/video_skeletons/fast_pose_duc_ResNet50_unshuffle/',
    f'{dir}/frames_skeletons/HRNet_HRNet_W32/',
    f'{dir}/video_skeletons/HRNet_HRNet_W32/',
    f'{dir}/frames_skeletons/fast_pose_dcn_ResNet50_dcn/',
    f'{dir}/video_skeletons/fast_pose_dcn_ResNet50_dcn/',
]


class VideoSkeletonizer:
    def __init__(self, cfg_path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

            self.execute_model_simple_baseline_ResNet50 = self.cfg["execute_model_simple_baseline_ResNet50"]
            self.execute_model_fast_pose_ResNet50 = self.cfg["execute_model_fast_pose_ResNet50"]
            self.execute_model_fast_pose_duc_ResNet50_unshuffle = self.cfg[
                "execute_model_fast_pose_duc_ResNet50_unshuffle"]
            self.execute_model_HRNet_HRNet_W32 = self.cfg["execute_model_HRNet_HRNet_W32"]
            self.execute_model_fast_pose_dcn_ResNet50_dcn = self.cfg["execute_model_fast_pose_dcn_ResNet50_dcn"]
            self.execute_model_fast_pose_duc_ResNet50 = self.cfg["execute_model_fast_pose_duc_ResNet50"]

            self.file_id_model_simple_baseline_ResNet50 = self.cfg["file_id_model_simple_baseline_ResNet50"]
            self.file_id_model_fast_pose_ResNet50 = self.cfg["file_id_model_fast_pose_ResNet50"]
            self.file_id_model_fast_pose_duc_ResNet50_unshuffle = self.cfg[
                "file_id_model_fast_pose_duc_ResNet50_unshuffle"]
            self.file_id_model_HRNet_HRNet_W32 = self.cfg["file_id_model_HRNet_HRNet_W32"]
            self.file_id_model_fast_pose_dcn_ResNet50_dcn = self.cfg["file_id_model_fast_pose_dcn_ResNet50_dcn"]
            self.file_id_model_fast_pose_duc_ResNet50 = self.cfg["file_id_model_fast_pose_duc_ResNet50"]

            self.model_simple_baseline_ResNet50 = self.cfg["model_simple_baseline_ResNet50"]
            self.model_fast_pose_ResNet50 = self.cfg["model_fast_pose_ResNet50"]
            self.model_fast_pose_duc_ResNet50_unshuffle = self.cfg["model_fast_pose_duc_ResNet50_unshuffle"]
            self.model_HRNet_HRNet_W32 = self.cfg["model_HRNet_HRNet_W32"]
            self.model_fast_pose_dcn_ResNet50_dcn = self.cfg["model_fast_pose_dcn_ResNet50_dcn"]
            self.model_fast_pose_duc_ResNet50 = self.cfg["model_fast_pose_duc_ResNet50"]

            self.cfg_model_simple_baseline_ResNet50 = self.cfg["cfg_model_simple_baseline_ResNet50"]
            self.cfg_model_fast_pose_ResNet50 = self.cfg["cfg_model_fast_pose_ResNet50"]
            self.cfg_model_fast_pose_duc_ResNet50_unshuffle = self.cfg["cfg_model_fast_pose_duc_ResNet50_unshuffle"]
            self.cfg_model_HRNet_HRNet_W32 = self.cfg["cfg_model_HRNet_HRNet_W32"]
            self.cfg_model_fast_pose_dcn_ResNet50_dcn = self.cfg["cfg_model_fast_pose_dcn_ResNet50_dcn"]
            self.cfg_model_fast_pose_duc_ResNet50 = self.cfg["cfg_model_fast_pose_duc_ResNet50"]

            self.file_id_to_process = self.cfg["file_to_process_id"]
            self.output_make_frames_folder = self.cfg["output_make_frames_folder"]
            self.output_folder = self.cfg["output_folder"]
            self.video_folder_path = self.cfg["video_folder_path"]

    def build(self):
        # Создаем директории
        for directory in frame_dirs:
            os.makedirs(directory, exist_ok=True)

    def clear_last_work_directory(self, model):
        output_make_frames_folder = self.output_make_frames_folder + model

        for filename in os.listdir(output_make_frames_folder):
            file_path = os.path.join(output_make_frames_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logging.info(f"Error: {e}")

    def video2frames(self, video_path: str, model_type: str):
        output_folder = self.output_folder

        # Шаг 1: Разбиваем видео на фреймы и сохраняем их в папку
        cap = cv2.VideoCapture(video_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_number += 1

        cap.release()

    def processed_video(self, model_type: str, model: str, video_path: str):
        self.clear_last_work_directory(model_type)

        command = f"python3 /home/student-2/skeletons/AlphaPose/scripts/demo_inference.py --cfg /home/student-2/skeletons/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint /home/student-2/skeletons/AlphaPose/pretrained_models/{model} --indir /home/student-2/skeletons/frames/ --outdir /home/student-2/skeletons/frames_skeletons/{model_type}/ --save_img"
        subprocess.call(command, shell=True)
        self.concatenate_processed_frames(model_type, video_path)

    def concatenate_processed_frames(self, model_type: str, video_path: str):
        # Укажите путь к папке с фреймами и выходному видеофайлу
        frames_folder = f'/home/student-2/skeletons/frames_skeletons/{model_type}/vis/'
        new_video_name = video_path.split('/')[-1]
        output_video_name = f'/home/student-2/skeletons/video_skeletons/{model_type}/{new_video_name}'

        # Получите список файлов фреймов
        frame_files = [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')]
        frame_files.sort()  # Сортируйте их, чтобы они шли в правильном порядке

        # Откройте первый фрейм, чтобы получить размер
        sample_frame = cv2.imread(frame_files[0])
        height, width, layers = sample_frame.shape

        # Укажите параметры видео (кодек, количество кадров в секунду и размер)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Количество кадров в секунду
        output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

        # Запись каждого фрейма в выходное видео
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            output_video.write(frame)

        # Закройте объект VideoWriter после завершения записи
        output_video.release()

    def video2processed_frames(self, video_path):
        if self.execute_model_simple_baseline_ResNet50:
            self.processed_video(model_type='simple_baseline_ResNet50', model=self.model_simple_baseline_ResNet50, video_path=video_path)

        if self.execute_model_fast_pose_ResNet50:
            self.processed_video(model_type='fast_pose_ResNet50', model=self.model_fast_pose_ResNet50, video_path=video_path)

        if self.execute_model_fast_pose_duc_ResNet50_unshuffle:
            self.processed_video(model_type='fast_pose_duc_ResNet50_unshuffle', model=self.model_fast_pose_duc_ResNet50_unshuffle, video_path=video_path)

        if self.execute_model_HRNet_HRNet_W32:
            self.processed_video(model_type='HRNet_HRNet_W32', model=self.model_HRNet_HRNet_W32, video_path=video_path)

        if self.execute_model_fast_pose_dcn_ResNet50_dcn:
            self.processed_video(model_type='fast_pose_dcn_ResNet50_dcn', model=self.model_fast_pose_dcn_ResNet50_dcn, video_path=video_path)

        if self.execute_model_fast_pose_duc_ResNet50:
            self.processed_video(model_type='fast_pose_duc_ResNet50', model=self.model_fast_pose_duc_ResNet50, video_path=video_path)

    def process_videos2frames(self):
        convert_videos_to_mp4(self.video_folder_path)
        for root, dirs, files in os.walk(self.video_folder_path):
            if len(files) < 0:
                continue
            for file in files:
                if file.endswith('.mp4'):
                    # print(root + file)
                    output_folder = self.output_folder
                    try:
                        self.video2frames(root + file, '')
                        self.video2processed_frames(root + file)
                    except Exception as e:
                        logging.info(f"Error: {e}")
                    for filename in os.listdir(output_folder):
                        file_path = os.path.join(output_folder, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            logging.info(f"Error: {e}")


def skeleton_process():
    vid_skeleton = VideoSkeletonizer('/home/student-2/skeletons/configs/skeleton_image.cfg')
    vid_skeleton.build()
    vid_skeleton.process_videos2frames()
    # vid_skeleton.concatenate_processed_frames(model_type='fast_pose_ResNet50', video_path='/home/student-2/skeletons/video_skeletons/fast_pose_ResNet50/new_video.mp4')
