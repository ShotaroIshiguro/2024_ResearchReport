# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

"""
指定された画像やビデオからデータを読み込み、顔の検出・クロップ・変換処理を行い、モデルに入力可能形式に変換
"""

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from . import detectors

# ビデオをフレームごとに画像として分解し、それぞれのフレームをjpegファイルとして保存
def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        # if count%sample_step == 0:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

# テストデータセットの準備
class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10, size=256, sort=False):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):  # testpathがリストだったらそのまま設定
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):   # testpathがフォルダだったら、全ての画像ファイルを取得し、設定
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']): # 個別画像ファイル場合はファイルをリストに変換し、設定
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):  # 動画の場合
            self.imagepath_list = video2sequence(testpath, sample_step) # フレームを画像として抽出し、リストに保存
        else:
            print(f'please check the test path: {testpath}')
            exit()
        # print('total {} images'.format(len(self.imagepath_list)))
        if sort:
            self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.size = size
        if face_detector == 'fan':  # 顔検出器Face Alignment Networkを初期化
            self.face_detector = detectors.FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    # データセットの総画像数
    def __len__(self):
        return len(self.imagepath_list)

    # 顔のバウンディングボックス情報から顔の中心とサイズを計算
    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    # 画像を読み込み、顔検出・クロップ処理を行い、正規化された顔画像と変換行列を取得
    def get_image(self, image):
        h, w, _ = image.shape
        # 顔のバウンディングボックスを取得
        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 4:
            print('no face detected! run original image')
            left = 0; right = h-1; top=0; bottom=w-1
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
        # bbox2point関数に左右上下情報をわたし、中心座標とサイズを計算
        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size*self.scale)
        # もともとの画像座標
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        # クロップ画像座標
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        # ソースからターゲットにクロップするための変換行列
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # 画像を正規化し、wrap関数で画像をクロップ・リサイズする
        image = image/255.
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # Pytorch用変換
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }

    # インデックスを指定して、データセット内の画像を取得し、顔のクロップ処理を適用して正規化された画像を返す
    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        # indexで指定された画像ファイルの名前部分をimagenameとして保存
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        im = imread(imagepath)

        if self.size is not None:
            # sizeが指定されている場合、読み込んだ画像をリサイズ
            im = (resize(im, (self.size, self.size), anti_aliasing=True) * 255.).astype(np.uint8)

        image = np.array(im)
        # グレースケールやαチャネルを持っている場合もRGB形式に変換
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            # 顔のクロップを行う場合は、顔のランドマークデータ(kpt)、バウンディングボックスを使ってクロップ領域を計算
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = os.path.splitext(imagepath)[0]+'.mat'
            kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
            if os.path.exists(kpt_matpath): # 顔のランドマークを示す.matファイルがある場合
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')  # クロップ領域の中心とサイズを決定
            elif os.path.exists(kpt_txtpath):   # 顔のランドマークを示す.txtフィルがある場合
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:   # 顔の検出器使用
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                # 顔検出器から得た情報からクロップ領域の中心サイズを計算
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        # DST_PTS = np.array([[0, 0], [0, h-1], [w-1, 0]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        # 変換行列を取得
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # 正規化し、warp関数で画像をクロップ・リサイズして、crop_sizeに整形
        image = image/255.
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }