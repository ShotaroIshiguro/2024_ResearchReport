import bz2
import os
import os.path as osp

import dlib
import numpy as np
import PIL.Image
import requests
import scipy.ndimage
from tqdm import tqdm
from argparse import ArgumentParser

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

"""
入力画像から顔のランドマークを検出し、顔を正確にアラインメント(整列)させた後、指定したサイズで保存する処理
Dlibを使用して顔のランドマークを検出し、FFHQデータセットで使用されたアラインメント手法を実装
"""

# 顔のアラインメント関数
# 顔のランドマークに基づいて顔を正確に位置合わせ、正方形の出力画像として保存
def image_align(src_file,               # アラインメントを行う元画像のファイルパス
                dst_file,               # アライメント後の画像を保存するためのパス
                face_landmarks,         # 顔のランドマークのリスト
                output_size=1024,
                transform_size=4096,
                enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    # 顔の向きを正しくそろえるための、左目・右目・口のランドマークを抽出
    lm = np.array(face_landmarks)
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise

    # Calculate auxiliary vectors.
    # 顔の回転や位置合わせに使用する基準点、ベクトルを計算
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    # 目と口の位置に基づき、顔を囲む四角形quadを定義
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print(
            '\nCannot find source image. Please run "--wilds" before "--align".'
        )
        return
    img = PIL.Image.open(src_file)
    img = img.convert('RGB')

    # Shrink.
    # 顔のサイズが出力サイズに対して大きすぎる場合は画像を縮小
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        # img = img.resize(rsize, PIL.Image.ANTIALIAS)
        img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    # 四角形quadに基づいて画像をクロップし、顔の領域だけ意を切り取る
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    # 顔が画像の枠外にはみ出す場合、パディングを追加して画像の境界部分を補完
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    if img.size[0] < output_size or img.size[1] < output_size:
        print(src_file + ' resolution too low!')
        return

    # Transform.
    # quadに基づく画像の回転・変形を行う、その後リサイズ
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        # img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    img.save(dst_file, 'PNG')

# 画像から顔のランドマークを検出
class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        # 顔検出器
        self.detector = dlib.get_frontal_face_detector(
        )  # cnn_face_detection_model_v1 also can be used
        # ランドマーク予測器　事前予測モデルを解凍して使用
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        # 顔検出
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [
                (item.x, item.y)
                # 68個の顔ランドマークを予測
                for item in self.shape_predictor(img, detection).parts()
            ]
            yield face_landmarks

# bz2形式で圧縮されたファイルを解凍
def unpack_bz2(src_path):
    dst_path = src_path[:-4]
    if os.path.exists(dst_path):
        print('cached')
        return dst_path
    data = bz2.BZ2File(src_path).read()
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

# ランドマークを使用したアラインメント処理
def work_landmark(raw_img_path, img_name, face_landmarks, output_size):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    if os.path.exists(aligned_face_path):
        return
    # 入力画像とランドマークを使用して、顔のアラインメント処理を行う
    image_align(raw_img_path,
                aligned_face_path,
                face_landmarks,
                output_size=output_size)


def get_file(src, tgt):
    if os.path.exists(tgt):
        print('cached')
        return tgt
    tgt_dir = os.path.dirname(tgt)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    file = requests.get(src)
    open(tgt, 'wb').write(file.content)
    return tgt


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    # コマンドライン引数を使って、ディレクトリや出力サイズを指定
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input_imgs_path",
                        type=str,
                        default="imgs",
                        help="input images directory path")
    parser.add_argument("-o",
                        "--output_imgs_path",
                        type=str,
                        default="imgs_align",
                        help="output images directory path")
    parser.add_argument("-s",
                        "--output_size",
                        type=int,
                        default=256,
                        help="output size of images")

    args = parser.parse_args()

    # Dlibのランドマークモデルをダウンロードし、解凍して使える状態にする
    landmarks_model_path = unpack_bz2(
        get_file(
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'temp/shape_predictor_68_face_landmarks.dat.bz2'))

    RAW_IMAGES_DIR = args.input_imgs_path
    ALIGNED_IMAGES_DIR = args.output_imgs_path
    output_size = args.output_size

    if not osp.exists(ALIGNED_IMAGES_DIR): os.makedirs(ALIGNED_IMAGES_DIR)

    files = os.listdir(RAW_IMAGES_DIR)
    print(f'total img files {len(files)}')
    with tqdm(total=len(files)) as progress:

        res = []
        # ランドマーク検出器を使える状態にする
        landmarks_detector = LandmarksDetector(landmarks_model_path)
        # 入力画像ディレクトリ内の画像ファイルを一つずつ処理し、顔のランドマークを検出後、
        # それを使って顔を整列させ、結果を出力ディレクトリに保存
        for img_name in files:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            for i, face_landmarks in enumerate(
                    landmarks_detector.get_landmarks(raw_img_path),
                    start=1):

                work_landmark(raw_img_path, img_name, face_landmarks, output_size)
                progress.update()

    print(f"output aligned images at: {ALIGNED_IMAGES_DIR}")
