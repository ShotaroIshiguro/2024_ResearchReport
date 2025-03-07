# DiffusionRig-Emo

By replacing DECA, which gives the physical conditions of DiffusionRig's face, with EMOCA, more realistic facial expression conversion is possible.  

*Shotaro Ishiguro*

![Image 1](https://github.com/ShotaroIshiguro/DiffusionRig_EMO-test-/blob/main/architecture.png)

## Setup & Preparation
```
conda env create -n DRE python=3.8 --file environment_DRE.yml
conda activate DRE
cd DiffusionRig_main
pip install -e .
```
If you encount the error with omegaconf, try these prompts in `DRE`.
```
pip install "pip<24.1"
pip install omegaconf==2.0.5 hydra-core==1.0.7
```
Cython may not be installed correctly, so install it separately.：
```
pip install Cython==0.29.14
```
If pytorch3d installation fails:
```
conda install pytorch3d -c pytorch3d
```

## Data Preparation
We use FFHQ and AffectNet to train the first stage and a personal photo album to train the second stage. Before training, you need to extract, with DECA or EMOCA, the physical buffers for those images.

### DECA Setup
Before doing data preparation for training, please first download the source files and checkpoints of DECA to set it up (you will need to create an account to download FLAME resources):

1. `deca_model.tar`:Visit [this page](https://github.com/yfeng95/DECA) to download the pretrained DECA model.
2. `generic_model.pkl`: Visit [this page](https://flame.is.tue.mpg.de/download.php) to download `FLAME 2020` and extract `generic_model.pkl`.
3. `FLAME_texture.npz`: Visit [this same page](https://flame.is.tue.mpg.de/login.php) to download the `FLAME texture space` and extract `FLAME_texture.npz`.
4. Download the other files listed below from [DECA's Data Page](https://github.com/yfeng95/DECA/tree/master/data) and put them also in the `data/` folder:

```
data/
  deca_model.tar
  generic_model.pkl
  FLAME_texture.npz
  fixed_displacement_256.npy
  head_template.obj
  landmark_embedding.npy
  mean_texture.jpg
  texture_data_256.npy
  uv_face_eye_mask.png
  uv_face_mask.png
```

### EMOCA Setup
You need to download and unzipu a few assets to create 3D shapes.  
assets directory will be created.

```
cd gdl_apps/EMOCA/demos
bash download_assets.sh
```

### Dataset for Stage1
Before training, pre-extract 3D facial features from facial images included in FFHQ and AffectNet.　　
The total iteration displayed in the tqdm bar is the number of target images divided by the batch size (default is 8 images).
```
cd DiffusionRig_main
# FFHQ(DECA)
python scripts/create_data.py --data_dir FFHQ/FFHQ_images \
    --output_dir ffhq256_deca.lmdb --image_size 256 --use_meanshape False \
    --use_model DECA
# FFHQ(EMOCA)
python scripts/create_data.py --data_dir FFHQ/FFHQ_images \
    --output_dir ffhq256_emoca.lmdb --image_size 256 --use_meanshape False \
    --use_model EMOCA
```

### Dataset for Stage2
Extract face alignment and physical buffer in advance for personal albums used in Stage 2.
```
conda deactivate DRE
conda env create -n DRE_align python=3.9 --file environment_DRE_align.yml
conda activate DRE_align

python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM \
    -o PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM -s 256

conda deactivate DER_align
conda activate DRE

# Personal_Album(DECA)
python scripts/create_data.py --data_dir PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    --output_dir NAME_MODEL.lmdb --image_size 256 --use_meanshape True --use_model DECA
# Personal_Album(EMOCA)
python scripts/create_data.py --data_dir PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    --output_dir NAME_MODEL.lmdb --image_size 256 --use_meanshape True --use_model EMOCA
```

## Training

### Stage1:Learning Generic Face Priors
Training to learn common facial features:
- Global encoder can be selected from resnet18 and resnet50.
- latent_dim is the dimension of the latent variable in the diffusion model.
- Use lmdb files to efficiently handle rendered images and physical buffers.
- Distributed learning using mpiexec is also possible.
- If you want to resume a training process, simply add `--resume_checkpoint PATH_TO_THE_MODEL`.
```
python scripts/train.py --latent_dim 64 --encoder_type resnet18  \
    --log_dir log/stage1/emoca_FFHQ_resnet18_batch16 --data_dir ffhq256_emoca.lmdb \
    --lr 1e-4 --p2_weight True --image_size 256 --batch_size 16 --max_steps 50000 \
    --num_workers 8 --save_interval 5000 --stage 1
```

### Stage 2: Learning Personalized Priors
Finetune the model on your tiny personal album:

```
python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log/stage2 --resume_checkpoint log/stage1/[MODEL_NAME].pt \
    --data_dir NAME_MODEL.lmdb --lr 1e-5 \
    --p2_weight True --image_size 256 --batch_size 4 --max_steps 5000 \
    --num_workers 8 --save_interval 5000 --stage 2
```

## Inference

Three elements can be edited based on the physical buffer: Exp, Pose, and Light

- Select the features of the target image (head orientation, facial expression, lighting) and transfer only those features to the source image.
- Source images should pick from the personal album used in stage2.
- Multiple images can be specified as source images by specifying the directory in `--source`.
- It is possible to specify whether the physical buffer of the target image and source image is obtained from DECA or EMOCA.
- If you want to use the model trained by the Global encoder with resnet18 version, you must change the encoder_type in `utils/script_util.py/def model_and_diffusion_defaults()`
- The code to select resnet18 or resnet50 at the time of inference has not been implemented. Please open `utils/scripts_util.py` and select the variable on lines 56 and 57 according to the resnet to be used.

```
python scripts/inference.py --source jisaku_training/Hitoshi_aligned/ \
   --modes exp --model_path log/stage2/stage2_model005000_Hitoshi.pt \
   --timestep_respacing ddim20 \
   --meanshape personal_deca_Hitoshi.lmdb/mean_shape.pkl \
   --target jisaku_training/obama_aligned/obama_12.png \
   --output_dir output_dir/target_smile_OBAMA12/targetEMOCA_sourseDECA \
   --target_model EMOCA \
   --source_model DECA
```

## 3D face shape acquisition using EMOCA

By executing `gdl_apps/EMOCA/demos/test_emoca_on_images.py`, you can obtain a 3D face model, latent code, rendering results, etc. from a 2D face image.

```
python demos/test_emoca_on_images.py --input_folder demos/test_images \
    --output_folder demos/output --model_name EMOCA_v2_lr_mse_20 \
    --save_mesh True --save_codes True
```

## References
1. DiffusionRig: Learning Personalized Priors for Facial Appearance Editing  
CVPR 2023  
https://arxiv.org/pdf/2304.06711  
https://github.com/adobe-research/diffusion-rig
```
@misc{ding2023diffusionriglearningpersonalizedpriors,
      title={DiffusionRig: Learning Personalized Priors for Facial Appearance Editing}, 
      author={Zheng Ding and Xuaner Zhang and Zhihao Xia and Lars Jebe and Zhuowen Tu and Xiuming Zhang},
      year={2023},
      eprint={2304.06711},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.06711}, 
}
```
2. EMOCA: Emotion Driven Monocular Face Capture and Animation  
CVPR 2022  
https://arxiv.org/pdf/2204.11312  
https://github.com/radekd91/emoca
```
@misc{danecek2022emocaemotiondrivenmonocular,
      title={EMOCA: Emotion Driven Monocular Face Capture and Animation}, 
      author={Radek Danecek and Michael J. Black and Timo Bolkart},
      year={2022},
      eprint={2204.11312},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.11312}, 
}
```

3. AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild  
IEEE Transactions on Affective Computing, 2017  
http://mohammadmahoor.com/affectnet/  
https://github.com/djordjebatic/AffectNet  
```
@ARTICLE{8013713,
    author={A. Mollahosseini and B. Hasani and M. H. Mahoor},
    journal={IEEE Transactions on Affective Computing},
    title={AffectNet: A Database for Facial Expression, Valence, and Arousal
    Computing in the Wild},
    year={2017},
    volume={PP},
    number={99},
    pages={1-1},}
```

4. Learning an Animatable Detailed 3D Face Model from In-The-Wild Images  
SIGGRAPH 2021  
https://arxiv.org/abs/2012.04012  
https://github.com/yfeng95/DECA  
```
@misc{feng2021learninganimatabledetailed3d,
      title={Learning an Animatable Detailed 3D Face Model from In-The-Wild Images}, 
      author={Yao Feng and Haiwen Feng and Michael J. Black and Timo Bolkart},
      year={2021},
      eprint={2012.04012},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2012.04012}, 
}
```
5. A Style-Based Generator Architecture for Generative Adversarial Networks(FFHQ)  
CVPR 2019 final version  
https://arxiv.org/pdf/1812.04948  
https://github.com/NVlabs/ffhq-dataset  
```
@misc{karras2019stylebasedgeneratorarchitecturegenerative,
      title={A Style-Based Generator Architecture for Generative Adversarial Networks}, 
      author={Tero Karras and Samuli Laine and Timo Aila},
      year={2019},
      eprint={1812.04948},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1812.04948}, 
}
```

### How to use git

- githubからSSHキーを取得
    git@github.com:ShotaroIshiguro/DiffusionRig_EMO-test-.git
- `git init`で追加したいファイルがあるローカルディレクトリにて受け入れ準備
- `git remote add origin SSH_KEY`
- `git status`で現状況確認(masterブランチにいたら`git branch -M main`でmainブランチに移動)
- `git fetch`でリモートから最新情報を取得
- `git pull origin main`でローカルのmainブランチに反映
- `git add ADD_FILE` or `git add ADD_FILE/`でステージング領域に追加
- `git status`でステージングを確認
- `git commit -m "コメント追加"`でコミット
- `git push origin main`でプッシュ






