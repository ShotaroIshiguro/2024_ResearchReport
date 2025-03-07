import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import scipy.io

# AFLW2000データセット(顔のランドマークを含む画像データセット)から読み込み、前処理を行い、
# モデルに入力可能な形式で画像を返すために使用
class AFLW2000(Dataset):
    # データセットの初期化を行い、画像ファイルのパスをリストとして保持
    def __init__(self, testpath='/ps/scratch/yfeng/Data/AFLW2000/GT', crop_size=224):
        '''
            data class for loading AFLW2000 dataset
            make sure each image has corresponding mat file, which provides cropping infromation
        '''
        if os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png')
        elif isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png']):
            self.imagepath_list = [testpath]
        else:
            print('please check the input path')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = 1.6
        self.resolution_inp = crop_size

    # データセットの画像数
    def __len__(self):
        return len(self.imagepath_list)

    # 指定インデックスの画像とその関連情報を取得し、前処理を行う
    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]  # 画像の名前
        image = imread(imagepath)[:,:,:3]   # 画像を読み込み、最初の3チャネルのみ使用(RGB画像として扱う)
        # scipyで対応する.matファイルからランドマークの3D座標データ(68個)を読み込む
        kpt = scipy.io.loadmat(imagepath.replace('jpg', 'mat'))['pt3d_68'].T    
        # ランドマークデータkptをもとに、顔のmax・min座標を計算して、顔領域を囲む矩形を求める    
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        # 矩形の中央を基準とし、サイズを計算し、self.scaleでスケールアップしてクロップ領域を定義
        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        size = int(old_size*self.scale)

        # crop image
        # ソース座標(クロップする矩形の角座標)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        # ターゲット座標(リサイズ後の画像の角の座標)
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        # ソースからターゲットに類似(単純)変換するための変換行列tform
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        # wrap関数を用いて、推定変換を逆適用することで、画像をリサイズ・クロップする
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # Pytroch用変換
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }