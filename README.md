# lle

测试EnGAN：
python benchmark/EnlightenGAN/scripts/script.py --predict
测试Colie：
python benchmark/colie/colie.py --alpha 1 --beta 20 --gamma 8 --delta 5

把FiveK从gt到input：
python datasets/generate_fivek_lowlight.py --gamma 5.0
测试（存在/preview）：
python datasets/generate_fivek_lowlight.py --dry_run --gamma 5.0

datastes links:
LSRW(Huawei/Nikon): https://opendatalab.com/OpenDataLab/LSRW
FiveK: https://www.kaggle.com/datasets/weipengzhang/adobe-fivek
CODaN: https://github.com/Attila94/CODaN
DarkFace: https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset
LIS: https://github.com/Linwei-Chen/LIS
