import argparse
import sys
import os
# 現在のスクリプトの親ディレクトリがパスに追加され、そこにあるモジュールがインポートできるようになる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.resample import create_named_schedule_sampler    # スケジュールサンプラーの作成関数
from utils import dist_util, logger
from utils.image_datasets import load_data, load_data_local     # LMDBデータからの読み込み
from utils.train_util import TrainLoop  # DiffusionRigのパラメータ更新
from utils.script_util import (
    model_and_diffusion_defaults,   # モデルのパラメータと、拡散プロセスのデフォルト設定を返す
    create_model_and_diffusion,     # modelとしてDiffusionRig、diffusionとして損失関数やスケジュールを定義したcreate_gaussian_diffusionを返す
    args_to_dict,
    add_dict_to_argparser,
)

"""
拡散モデルを使用したトレーニングを実施
"""

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log(args)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    # 拡散過程の時間ステップtをどのようにサンプリングするか決定する、トレーニングのサンプラーを宣言
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # FFHQ or 個人アルバムの学習かで、使用するデータローダが異なる
    if args.stage == 1:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.stage == 2:       # 
        data = load_data_local(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
        )

    # トレーニング開始　utils.train_util.pyのTrainloopを使用
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        stage=args.stage,
        max_steps=args.max_steps,
        auto_scale_grad_clip=args.auto_scale_grad_clip,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        batch_size=1,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        log_dir="stage1",
        num_workers=16,
        max_steps=0,
        auto_scale_grad_clip=1.0,
        stage=1,
    )
    # モデルのパラメータと、拡散プロセスのデフォルト設定を取得し、defaultパーサーに追加
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
