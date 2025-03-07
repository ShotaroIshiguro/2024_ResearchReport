import os, sys
from tqdm import tqdm
import torch as th

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

import pickle
from torch.utils.data import DataLoader
import lmdb
from PIL import Image
from io import BytesIO
import pickle
import argparse
from utils.script_util import (
    add_dict_to_argparser,
)

import gdl
from pathlib import Path
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
import pickle
from gdl_apps.EMOCA.utils.load import load_model

"""
DECA(Deep 3D Face Analysis)を使って3D顔モデルの特徴を抽出し、
その結果をLMDB(Lightning Memory-Mapped Database)に保存
LMDBは高パフォーマンスなデータベースで、大量のバイナリデータ(画像や3Dモデルなどを効率よく保存するために使用
"""


def main():
    # コマンドライン引数を取得し、DECAモデルを初期化
    args = create_argparser().parse_args()

    # Build DECA
    # テクスチャマッピングを有効化し、FLAMEテクスチャモデルを使って顔の詳細なレンダリングを行う
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca = DECA(config=deca_cfg, device="cuda")

    # EMOCA改変 #########################################
    use_model = args.use_model
    print("モード：" + str(use_model))
    EMOCA_path_to_models = str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
    EMOCA_model_name = 'EMOCA_v2_lr_mse_20'
    EMOCA_mode = 'detail'
    emoca, conf = load_model(EMOCA_path_to_models, EMOCA_model_name, EMOCA_mode)
    emoca.cuda()
    emoca.eval()
    #####################################################

    # Create Dataset
    # コマンドライン引数data_dirからローカルデータセットを読み込み
    dataset_root = args.data_dir
    testdata = datasets.TestData(
        # 顔の部分のみクロップ
        dataset_root, iscrop=True, size=args.image_size, sort=True
    )
    # バッチサイズ指定
    batch_size = args.batch_size
    loader = DataLoader(testdata, batch_size=batch_size)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # meanshapeを使用する場合、データセット内の各画像の形状情報を取得し、それを平均化して使用
    if args.use_meanshape:
        shapes = []
        for td in testdata:
            img = td["image"].to("cuda").unsqueeze(0)
            # DECAのencode関数で、画像から3Dshapeを抽出し、それらを平均化して保存
            if use_model == "DECA":
                code = deca.encode(img)
            elif use_model == "EMOCA":
                code, _, _ == test(emoca, img)
            shapes.append(code["shape"].detach())
        mean_shape = th.mean(th.cat(shapes, dim=0), dim=0, keepdim=True)
        with open(os.path.join(output_dir, "mean_shape.pkl"), "wb") as f:
            pickle.dump(mean_shape, f)

    # LMDBデータベースに接続し、データの書き込み
    with lmdb.open(output_dir, map_size=1024**4, readahead=False) as env:
        total = 0
        for batch_id, data in enumerate(tqdm(loader)):

            with th.no_grad():
                inp = data["image"].to("cuda")
                # 各バッチに対して、まず入力画像をDECAモデルにエンコードし、顔の形状、表情、ポーズなどを抽出
                if use_model == "DECA":
                    codedict = deca.encode(inp)
                elif use_model == "EMOCA":
                    codedict, _, _ = test(emoca, inp)
                # 変換行列取得
                tform = data["tform"]
                tform = th.inverse(tform).transpose(1, 2).to("cuda")
                original_image = data["original_image"].to("cuda")

                if args.use_meanshape:
                    codedict["shape"] = mean_shape.repeat(inp.shape[0], 1)
                codedict["tform"] = tform

                # DECAのデコードで顔情報から3Dモデルを復元し、オブジェクト情報やレンダリング結果をopdictに格納
                opdict, _ = deca.decode(
                    codedict,
                    render_orig=True,
                    original_image=original_image,
                    tform=tform,
                )
                opdict["inputs"] = original_image

                # レンダリングされたアルベド画像・法線画像・レンダリング結果・元画像を順にLMDBデータベースに保存
                for item_id in range(inp.shape[0]):
                    i = batch_id * batch_size + item_id

                    image = (
                        (original_image[item_id].detach().cpu().numpy() * 255)
                        .astype("uint8")
                        .transpose((1, 2, 0))
                    )
                    image = Image.fromarray(image)

                    albedo_key = f"albedo_{str(i).zfill(6)}".encode("utf-8")
                    # BytesIO を使用して、画像データをバイナリ形式に変換し、LMDBに保存できる形式へ
                    buffer = BytesIO()
                    pickle.dump(opdict["albedo_images"][item_id].detach().cpu(), buffer)
                    albedo_val = buffer.getvalue()

                    normal_key = f"normal_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(opdict["normal_images"][item_id].detach().cpu(), buffer)
                    normal_val = buffer.getvalue()

                    rendered_key = f"rendered_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    pickle.dump(
                        opdict["rendered_images"][item_id].detach().cpu(), buffer
                    )
                    rendered_val = buffer.getvalue()

                    image_key = f"image_{str(i).zfill(6)}".encode("utf-8")
                    buffer = BytesIO()
                    image.save(buffer, format="png", quality=100)
                    image_val = buffer.getvalue()

                    with env.begin(write=True) as transaction:
                        transaction.put(albedo_key, albedo_val)
                        transaction.put(normal_key, normal_val)
                        transaction.put(rendered_key, rendered_val)
                        transaction.put(image_key, image_val)
                    total += 1
        # データベースに保存された総データ数を記録
        with env.begin(write=True) as transaction:
            transaction.put("length".encode("utf-8"), str(total).encode("utf-8"))

# コマンドライン引数を定義
def create_argparser():
    defaults = dict(
        data_dir="",
        output_dir="",
        image_size=256,
        batch_size=8,
        use_meanshape=False,    # 各画像の形状平均化せず
        use_model = "",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
