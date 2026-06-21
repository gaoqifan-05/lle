# lle

把FiveK从gt到input：
python datasets/generate_fivek_lowlight.py --gamma 5.0
测试（存在/preview）：
python datasets/generate_fivek_lowlight.py --dry_run --gamma 5.0

测试benchmark：run_benchmark.py
python run_benchmark.py --model all --dataset all
全模型×全数据集
python run_benchmark.py --model all --dataset all
只算指标不推理
python run_benchmark.py --model all --dataset all --skip_inference
单模型×单数据集，自定义降采样
python run_benchmark.py --model colie --dataset fivek --resize_size 256
CoLIE 自定义参数
python run_benchmark.py --model colie --dataset fivek --colie_alpha 0.5 --colie_beta 0.2

datastes links:
LSRW(Huawei/Nikon): https://opendatalab.com/OpenDataLab/LSRW
FiveK: https://www.kaggle.com/datasets/weipengzhang/adobe-fivek
CODaN: https://github.com/Attila94/CODaN
DarkFace: https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset
LIS: https://github.com/Linwei-Chen/LIS
