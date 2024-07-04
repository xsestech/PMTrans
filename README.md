# PMTrans
Patch-Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective

### Pretrained Swin-B

- Скачайте [swin_base_patch4_window7_224_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and put it into `pretrained_models`
  

### Устанвока
- Установите `CUDA==12.4` с `cudnn9` [инстуркция](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Конда:
```bash
conda create -f environment.yml
conda activate torch
```
- Установите `Apex`:

```bash

cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Datasets:

- Скачайте `Office31, Office Home, VisDA and Domainnet` Пропишите пути в соответвующих файлах в `datasets/office31/`

 ```bash
  $ tree data
  datasets
  ├── ofice_home
  │   ├── Art.txt
  │   ├── Clipart.txt
  │   ├── Product.txt
  │   ├── Real_World.txt
  └── ...
  ```   

### Обучение:
```
python -m torch.distributed.run --nproc_per_node 1 --master_port=3011 dist_pmTrans.py --use-checkpoint --source dslr --target webcam --dataset office31  --tag PM --local_rank 0 --batch-size 32 --head_lr_ratio 10 --log-dir loghaha --output result --cfg configs/swin_base.yaml --data-root-path datasets/```
