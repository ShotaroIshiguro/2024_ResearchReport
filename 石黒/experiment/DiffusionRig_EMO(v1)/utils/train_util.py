import copy
import functools
import os
from utils import dist_util

import blobfile as bf
import torch as th
import torch.distributed as dist
# モデルを複数のGPUで並列処理するために、PytorchのDDPを使用し、分散環境でのパラメータ同期をサポート
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam

from . import logger
from diffusion.resample import LossAwareSampler, UniformSampler

import time
import datetime

from torch.cuda.amp import autocast, GradScaler

"""
PyTorchを使った分散トレーニングのためのトレーニングループを実装
チェックポイントの保存や再開、勾配のクリッピング、AMP (Automatic Mixed Precision) によるトレーニング効率化など
画像と物理的条件を用いてパラメータ更新
更新されるパラメータには、Diffusion中のUNetに含まれるconvの他、diffusion.model.DiffusionRig中にある
Global_latent_codeを訓練するためのResNetに含まれるconvなどもOK
"""

class TrainLoop:
    def __init__(
        self,
        *,
        model,      # 拡散モデル
        diffusion,  # 拡散過程
        data,
        batch_size,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
        stage=1,
        max_steps=0,
        auto_scale_grad_clip=1.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.stage = stage
        self.max_steps = max_steps
        self.auto_scale_grad_clip = auto_scale_grad_clip

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        # デフォルトのスケジュールサンプラーは均一サンプリング
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        if stage == 2:
            for n, p in self.model.named_parameters():
                if not n.startswith('denoisingUNet.'):
                    p.requires_grad = False

        self.opt = Adam(
            self.model.parameters(),
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        self.scaler = GradScaler()

        if self.resume_step:
            self._load_optimizer_state()

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model


        self.start_time = time.time()
        self.step_time = 0
        self.last_step_time = None

    # チェックポイントからモデルパラメータをロード、DDP(複数GPU)を使った分散トレーニングにおける同期もサポート
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            if self.stage == 1:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            ckpt = dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )

            self.model.load_state_dict(ckpt)

        dist_util.sync_params(self.model.parameters())
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    # メインループ、各ステップでバッチデータを取得し、run_stepメソッドで順・逆伝播を実行
    def run_loop(self):
        while (
            not self.max_steps or
            self.step + self.resume_step < self.max_steps
        ):
            # データ取得、画像・物理条件を設定
            batch = next(self.data)

            image = batch['image']
            rendered, normal, albedo = batch['rendered'], batch['normal'], batch['albedo']
            physic_cond = th.cat([rendered, normal, albedo], dim=1)

            self.run_step(image, physic_cond)

            if (self.step + self.resume_step) % self.log_interval == 0:
                logger.log(logger.get_dir())
                logger.dumpkvs()
            if (self.step + self.resume_step) % self.save_interval == 0 and self.step > 0:
                self.save()
                logger.log(logger.get_dir())
                logger.dumpkvs()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step + self.resume_step - 1) % self.save_interval != 0:
            self.save()

    # 1ステップ分の処理
    # forward_backward関数の実行
    # Pytorchのモデル性能を向上させるための機能であるAMPを更新するためにGradScalerを用いる
    # stepメソッドでオプティマイザーのパラメータを更新
    def run_step(self, image, physic_cond):
        self.forward_backward(image, physic_cond)

        self.scaler.unscale_(self.opt)
        th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.auto_scale_grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()

        self.log_step()

    # 順・逆伝播の実行
    # 指定された物理条件と画像に対する損失を計算
    # 各ステップで損失にスケジュールサンプラーの重みを掛け合わせた平均をとり、逆伝播
    def forward_backward(self, image, physic_cond):
        self.opt.zero_grad()

        image = image.to(dist_util.dev())
        physic_cond = physic_cond.to(dist_util.dev())

        t, weights = self.schedule_sampler.sample(image.shape[0], dist_util.dev())

        # diffusion.training_lossesを呼び出すことで、物理的条件と画像に対する損失を計算
        # diffusion.gaussian_diffusion.pyのtraining_lossesを利用(functools.partialでモデル、画像、タイムステップ、物理条件を渡す)
        # 予測ノイズとガウスノイズのMSE損失
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model, # = model.DiffusionRig
            image,
            t,
            model_kwargs={'physic_cond': physic_cond, 'x_start': image},
        )

        with autocast():
            # with self.ddp_model.no_sync():
            losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # 損失の重み月平均を計算
            loss = (losses["loss"] * weights).mean()

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        # 計算した損失を用いたバックプロバケーション
        self.scaler.scale(loss).backward()

    def log_step(self):
        # 現在のトレーニングステップ数
        logger.logkv("step", self.step + self.resume_step)
        # 処理済みのサンプル累計数
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # トレーニング開始からの経過時間
        logger.logkv("time elapsed", str(datetime.timedelta(seconds=time.time() - self.start_time)))
        ct = time.time()
        if self.step_time == 0:
            self.step_time = ct - self.start_time
            self.last_step_time = ct
        else:
            self.step_time = 0.1 * self.step_time + 0.9 * (ct - self.last_step_time)
            self.last_step_time = ct
        # 次の10000ステップにかかる推定時間
        logger.logkv("time est.(10k)", str(datetime.timedelta(seconds=10000 * self.step_time)) )
        
        steps_to_go = (self.save_interval - self.step - self.resume_step) % self.save_interval
        # 次のチェックポイント保存までの推定時間
        logger.logkv("time est.(next ckpt)", str(datetime.timedelta(seconds=steps_to_go * self.step_time)))

    def save(self):
        def save_checkpoint(rate, params=None):

            state_dict = self.model.state_dict()
            
            if dist.get_rank() == 0:
                # state_dict = self.model.state_dict()
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0)

        if dist.get_rank() == 0 and self.stage == 1:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def log_loss_dict(diffusion, ts, losses):
    # logger.logkv_mean('loss', losses['loss'].mean().item())
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
