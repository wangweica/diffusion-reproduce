import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import scipy.io as sio
import numpy as np
import traceback
from tif_train_func import train
from natsort import natsorted

# 1. 数据路径配置
data_path = r"D:\studio\oct_gen\Code\Code\Code\FourD_INR\data"

# 2. 模型保存路径
model_save_path = r"D:\studio\oct_gen\models\all_inr"

# 3. 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Windows 多进程保护 ---
if __name__ == '__main__':
    # 检查并确保模型目录存在
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # 扫描数据目录中的所有 .mat 文件
    files = natsorted(os.listdir(data_path))
    
    infos = []
    for f in files:
        if f.endswith('.mat'):
            infos.append({
                "path": os.path.join(data_path, f),
                "name": f
            })
    
    print(f"找到待处理文件共: {len(infos)} 个")

    for i, info in enumerate(infos):
        # 加载数据
        mat = sio.loadmat(info['path'])
        
        # 4. 适配数据 Key 值 (根据你之前的调试，实际为 'img')
        if 'img' in mat:
            data = mat['img']
            # 检查如果数据是 0-255，则压缩到 0-1
            if data.max() > 1.0:
                data = data.astype(np.float32) / 255.0
        else:
            print(f"警告: 文件 {info['name']} 中未找到 'img' 变量，跳过。")
            continue
            
        # 5. 维度对齐：将 (64, 224, 224) 扩展为 (1, 64, 224, 224) 以适配 4D 模型
        if len(data.shape) == 3:
            data = data[np.newaxis, ...]
        
        print(f"--- 进度 [{i+1}/{len(infos)}] ---")
        print(f"开始训练: {info['name']} | 维度: {data.shape}")
        
        # 6. 调用训练函数
        try:
            train(
                exp_name="HyperINR-" + info['name'].replace('.mat', ''),
                device=device,
                img=data,
                model_save_path=model_save_path
            )
        except Exception as e:
            print(f"❌ 训练文件 {info['name']} 时出错:")
            traceback.print_exc()
            continue

    print("所有任务已执行完毕！")