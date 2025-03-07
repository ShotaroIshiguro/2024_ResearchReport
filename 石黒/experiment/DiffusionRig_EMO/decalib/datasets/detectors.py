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

import numpy as np
import torch

"""
顔のランドマークやバウンディングボックスを検出するためのクラス
"""

# face_alignmentライブラリを用いて、入力画像の顔ランドマークを検出し、顔のバウンディングボックスを返す
class FAN(object):
    def __init__(self):
        import face_alignment
        # 2Dランドマーク検出モデルを初期化
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        # 顔のランドマークを検出
        out = self.model.get_landmarks(image)
        if out is None:
            # 検出されなかった場合は、[0](検出失敗)と'kpt68'を返す
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()  # ランドマークの座標群
            # 顔全体の境界を特定
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

# facenet-pytorchライブラリのMTCNNを用いて、顔のバウンディングボックスを検出
class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)   # 画像内の全ての顔が検出対象
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        # 画像から顔のバウンディングボックスを検出
        out = self.model.detect(input[None,...])    # input[None, ...]で次元を追加することで4Dテンソルに変換し、期待入力形状に変換
        if out[0][0] is None:
            return [0]  # 顔の検出なし
        else:
            bbox = out[0][0].squeeze()  # バウンディングボックスをそのまま出力
            return bbox, 'bbox'



