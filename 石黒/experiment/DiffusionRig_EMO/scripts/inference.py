import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
from glob import glob

from utils.script_util import (
    model_and_diffusion_defaults,   # モデルのパラメータと、拡散プロセスのデフォルト設定を返す
    create_model_and_diffusion,     # modelとしてDiffusionRig、diffusionとして損失関数やスケジュールを定義した
                                    # create_gaussian_diffusionを返す
    add_dict_to_argparser,
    args_to_dict,
)


from torchvision.utils import save_image
from gdl_apps.EMOCA.utils.load import load_model
from decalib.deca import DECA   # FLAMEパラメータ、デコード結果格納
from decalib.utils.config import cfg as deca_cfg    # decaの設定、事前訓練済みモデルのパス指定など
from decalib.datasets import datasets as deca_dataset
# 指定された画像やビデオからデータを読み込み、顔の検出・クロップ・変換処理を行い、モデルに入力可能形式に変換

import gdl
from pathlib import Path
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
import pickle

"""
推論用
DECA)3D顔解析とレンダリング)と拡散モデルを組み合わせて、顔の画像を操作・生成するパイプラインを実装
2枚の画像)ソースとターゲット)の顔の特徴を組み合わせて、様々なモード)ポーズ、ライティング、表情など)での画像操作や生成を行う
ターゲット画像img2の特徴をソース画像リストに移植→ソース画像img1の表情だけ変更などが可能
ソース画像の個人アルバムで第二段階学習が必要←ソースのアイデンティティはいじらないから
latent modeはソース画像の顔は変わらないが、背景や髪がターゲット画像のものになる
"""

# DECAを使用して、ソース画像とターゲット画像の顔の特徴を抽出し、モード(ポーズ、ライティング、表情)で変形や操作を行う
def create_inter_data(dataset, modes, meanshape_path="", target_model=None, source_model=None):


    # Build DECA
    # DECAを初期化し、FLAMEを使うことや、どの学習済みパラメータを使うかどうかを指定
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg)

    # Build EMOCA
    # EMOCA改変
    EMOCA_path_to_models = str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
    EMOCA_model_name = 'EMOCA_v2_lr_mse_20'
    EMOCA_mode = 'detail'
    emoca, conf = load_model(EMOCA_path_to_models, EMOCA_model_name, EMOCA_mode)
    emoca.cuda()
    emoca.eval()


    # meanshapeファイルがあれば、それを読み込んで顔の形状を平衡化
    meanshape = None
    if os.path.exists(meanshape_path):
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        print("not use meanshape")

    # ターゲット画像img2(データセットの最後の画像)をdecaエンコードすることで3D顔特徴を抽出
    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    with th.no_grad():
        # ターゲット画像の特徴情報である形状・表情・姿勢パラメータを格納
        # print(type(img2))
        # print(img2.shape)
        ################################################# target EMOCA or DECA
        if target_model == "EMOCA":
            code2, _, _  = test(emoca, img2)
        elif target_model == "DECA":
            code2 = deca.encode(img2)
        
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")

    # 各ソース画像(img1)をエンコードし、3D顔特徴を取得し、後でターゲット画像の特徴と組み合わせることで変形
    for i in range(len(dataset) - 1):
        print(str(i+1) + "/" + str(len(dataset)-1) + "枚目生成中・・・")
        img1 = dataset[i]["image"].unsqueeze(0).to("cuda")

        with th.no_grad():
            ################################################## source EMOCA or DECA
            if source_model == "EMOCA":
                code1, _, _  = test(emoca, img1)
            elif source_model == "DECA":
                code1 = deca.encode(img1)
            # print("code1")
            # print(type(code1))
            # print(code1.shape)

        # ソース画像の顔をターゲット画像の顔と合わせるため、ポーズの中心位置を調整
        ffhq_center = None
        ffhq_center = deca.decode(code1, return_ffhq_center=True)

        tform = dataset[i]["tform"].unsqueeze(0)
        tform = th.inverse(tform).transpose(1, 2).to("cuda")
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        code1["tform"] = tform
        if meanshape is not None:
            code1["shape"] = meanshape

        # modesで指定された変形モードを適用
        # 各モードごとにcode1の顔特徴をコピーしてcodeに格納
        for mode in modes:
            code = {}
            codes_selection = ["shape", "tex", "exp", "pose", "cam", "light", "images", "detail", "tform"]
            for k in code1:     # ソース画像の顔特徴をcodeに格納
                # print(k)
                if k in codes_selection:
                    code[k] = code1[k].clone()

            origin_rendered = None

            # モードに合わせて、ソース画像のパラをターゲット画像のパラに置換
            if mode == "pose":
                code["pose"][:, :3] = code2["pose"][:, :3]
            elif mode == "light":
                code["light"] = code2["light"]
            elif mode == "exp":
                code["exp"] = code2["exp"]
                code["pose"][:, 3:] = code2["pose"][:, 3:]
            elif mode == "latent":
                pass

            # DECAによるデコードと画像生成
            # align_ffhqで顔の中心を合わせる
            # opdictに再構成画像や、法線マップ、アルベド画像を格納
            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
                align_ffhq=True,
                ffhq_center=ffhq_center,
            )

            # 生成された画像や法線マップ、アルベドマップ、モードの情報をバッチ形式で返す
            origin_rendered = opdict["rendered_images"].detach()
            batch = {}
            batch["image"] = original_image * 2 - 1                     # ソース画像
            batch["image2"] = image2 * 2 - 1                            # ターゲット画像
            batch["rendered"] = opdict["rendered_images"].detach()      # レンダリングされた画像
            batch["normal"] = opdict["normal_images"].detach()          # 法線マップ
            batch["albedo"] = opdict["albedo_images"].detach()          # アルベド画像
            batch["mode"] = mode
            batch["origin_rendered"] = origin_rendered
            yield batch

        if i+1 == len(dataset)-1:
            if target_model == "EMOCA":
                print("ターゲット：EMOCA")
            elif target_model == "DECA":
                print("ターゲット：DECA")
            if source_model == "EMOCA":
                print("ソース　　：EMOCA")
            elif source_model == "DECA":
                print("ソース　　：DECA")
            


# DECAと拡散モデルを組み合わせて、画像生成・変換を行うための手続きを実行
def main():
    args = create_argparser().parse_args()
    """
    source: ソース画像のパス
    target: ターゲット画像のパス
    output_dir: 生成された画像を保存するディレクトリ
    model_path: 学習済みのモデルのパス
    modes: 適用するモード(pose, exp, light など）"""

    print("creating model and diffusion...")
    # DiffusionRigとgaussian_diffusionプロセスを初期化
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 学習済みのDiffusionRigヲ読み込み、推論モードに移行
    ckpt = th.load(args.model_path)
    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()

    imagepath_list = []

    # ソース画像とターゲット画像を確認
    if not os.path.exists(args.source) or not os.path.exists(args.target):
        print("source file or target file doesn't exists.")
        return

    # ソース画像がディレクトリの場合は、全ての画像をリストに追加し、ターゲット画像を最後にリスト追加
    imagepath_list = []
    if os.path.isdir(args.source):
        imagepath_list += (
            glob(args.source + "/*.jpg")
            + glob(args.source + "/*.png")
            + glob(args.source + "/*.bmp")
        )
    else:
        imagepath_list += [args.source]
    imagepath_list += [args.target]
    # decalib.datasets.datasets.pyのTestDataクラスで、顔検出やクロップ処理を行う
    dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)

    # ソース・ターゲット画像から3D顔の情報を抽出し、指定モードに基づいた変形した画像データなどを返す
    # バッチ形式でレンダリング画像、法線マップ、アルベドマップがdataに格納
    modes = args.modes.split(",")
    data = create_inter_data(dataset, modes, args.meanshape, args.target_model, args.source_model)

    # 拡散プロセスによるサンプリングを行う関数を選択、通常はDDIM
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    os.system("mkdir -p " + args.output_dir)

    # 生成プロセスに使用するランダムノイズを生成
    noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

    vis_dir = args.output_dir
    idx = 0
    # 画像生成・保存
    for batch in data:
        image = batch["image"]      # ソース画像
        image2 = batch["image2"]    # ターゲット画像
        # レンダリング画像、法線マップ、アルベドマップ
        rendered, normal, albedo = batch["rendered"], batch["normal"], batch["albedo"]
        # 物理条件として格納
        physic_cond = th.cat([rendered, normal, albedo], dim=1)

        image = image
        physic_cond = physic_cond

        # latentモードが指定されると、ターゲット画像のglobal latent codeを使用
        # latentが指定されていないと、ソース画像の背景をそのまま用いる
        with th.no_grad():
            if batch["mode"] == "latent":
                detail_cond = model.encode_cond(image2) # Global latent code用のResNet
            else:
                detail_cond = model.encode_cond(image)

        # サンプリング(画像生成)を行う
        # 拡散モデルDiffusionRigでサンプリング関数としてDDImを使用
        sample = sample_fn(
            model,      # DiffusionRig
            (1, 3, args.image_size, args.image_size),   # 生成画像の形状
            noise=noise,        # ランダムノイズを入力として使用し、生成のスタートとする
            # DiffusionRigによる生成時にノイズ除去された画像をクリップするかどうかを指定
            clip_denoised=args.clip_denoised,
            # 物理的条件と顔以外の特徴をモデルに渡す
            model_kwargs={"physic_cond": physic_cond, "detail_cond": detail_cond},
        )
        # 生成画像の正則化([-1, 1]から[0, 1]へ)
        sample = (sample + 1) / 2.0
        sample = sample.contiguous()    # メモリは位置を連続化

        # 生成画像sampleを生成モードの名前とともに保存
        save_image(
            sample, os.path.join(vis_dir, "{}_".format(idx) + batch["mode"]) + ".png"
        )
        idx += 1


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="",
        source="",
        target="",
        output_dir="",
        modes="pose,exp,light",
        meanshape="",
        target_model="",
        source_model="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
