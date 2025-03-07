import os
from PIL import Image

# 入力ディレクトリパス
input_dir = 'jisaku_training/hayata_target'
# 出力ディレクトリパス
output_dir = 'jisaku_training/hayata_source'
# 新しいサイズ
new_size = (250, 250)  # (幅, 高さ)

# 入力ディレクトリ内のすべてのファイルを取得
files = os.listdir(input_dir)

# 画像ファイルのみを選択する
image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 出力ディレクトリが存在しない場合は作成する
os.makedirs(output_dir, exist_ok=True)

# すべての画像をリサイズする
for image_file in image_files:
    # 画像のフルパス
    image_path = os.path.join(input_dir, image_file)
    # 画像を開く
    img = Image.open(image_path)
    # 画像のリサイズ
    resized_img = img.resize(new_size, Image.LANCZOS)  # LANCZOSフィルタを使用してリサイズする
    # 出力ファイルパス
    output_path = os.path.join(output_dir, image_file)
    # リサイズされた画像を保存する
    resized_img.save(output_path)

print(f"All images resized and saved to {output_dir}.")
