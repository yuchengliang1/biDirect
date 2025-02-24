# meg数据 mat中data (64x7500x300 double) labels (300x1) double
# eeg数据 (300, 64, 7500) (300,)
import numpy as np
import scipy.io
import torch
import os
from torch.utils.data import Dataset  # 导入Dataset类，以防在同一个文件中定义EEGfNIRSDataset

# --- 定义文件路径 ---
eeg_data_path = r'D:\ESWA-BiDirect\TSMMF-ESWA\Data\ycl\eeg\all_epochs.npy'
eeg_labels_path = r'D:\ESWA-BiDirect\TSMMF-ESWA\Data\ycl\eeg\all_labels.npy'
meg_data_path = r'D:\ESWA-BiDirect\TSMMF-ESWA\Data\ycl\meg\yclSingle.mat'  # 假设MEG数据和标签都在这个.mat文件中


# 定义一个函数来加载所有数据，并返回处理后的PyTorch Tensor
def load_eeg_and_meg_data():
    """
    加载EEG和MEG数据，并进行必要的预处理和转换为PyTorch Tensor。
    MEG数据在此处被视为NIRS数据，以匹配EEGfNIRSDataset的命名习惯。

    Returns:
        tuple: 包含 (eeg_data_tensor, nirs_data_tensor, final_labels_tensor)
               如果加载失败，对应的Tensor可能为None。
    """
    eeg_data_tensor = None
    eeg_labels_tensor = None
    nirs_data_tensor = None  # 将MEG数据命名为nirs_data_tensor
    nirs_labels_tensor = None
    final_labels_tensor = None

    # --- 1. 读取EEG数据 ---
    # print("--- 尝试读取EEG数据 ---") # 导入时不打印，只在直接运行时打印
    try:
        eeg_data_np = np.load(eeg_data_path)
        eeg_labels_np = np.load(eeg_labels_path) - 1

        # 转换为PyTorch Tensor
        eeg_data_tensor = torch.tensor(eeg_data_np[:, :, 2000:4000], dtype=torch.float32)
        eeg_labels_tensor = torch.tensor(eeg_labels_np, dtype=torch.long)  # 标签通常是long类型

    except FileNotFoundError:
        print(f"错误：EEG文件未找到。请检查路径：{eeg_data_path} 或 {eeg_labels_path}")
    except Exception as e:
        print(f"读取EEG数据时发生错误：{e}")

    # --- 2. 读取MEG数据 (作为NIRS数据处理) ---
    # print("--- 尝试读取MEG数据 (作为NIRS) ---") # 导入时不打印
    try:
        meg_mat_content = scipy.io.loadmat(meg_data_path)

        meg_data_np = meg_mat_content['data'][:,2000:4000,:]
        meg_labels_np = meg_mat_content['labels'] - 1

        # 转置MEG数据以匹配 (样本数, 通道数, 时间点数)
        # 原始: (通道数, 时间点数, 样本数) -> (64, 7500, 300)
        # 目标: (样本数, 通道数, 时间点数) -> (300, 64, 7500)
        nirs_data_np_transposed = np.transpose(meg_data_np, (2, 0, 1))

        # 确保标签是1D数组，如果它是 (300, 1)
        nirs_labels_np_flat = meg_labels_np.flatten()

        # 转换为PyTorch Tensor
        nirs_data_tensor = torch.tensor(nirs_data_np_transposed, dtype=torch.float32)
        nirs_labels_tensor = torch.tensor(nirs_labels_np_flat, dtype=torch.long)

    except FileNotFoundError:
        print(f"错误：MEG文件未找到。请检查路径：{meg_data_path}")
    except KeyError as e:
        print(f"错误：.mat文件中缺少预期的键。请检查变量名 'data' 或 'labels'：{e}")
    except Exception as e:
        print(f"读取MEG数据时发生错误：{e}")

    # --- 3. 处理标签一致性 ---
    # 优先使用EEG标签，如果EEG标签不存在则使用NIRS标签
    if eeg_labels_tensor is not None and nirs_labels_tensor is not None:
        if torch.equal(eeg_labels_tensor, nirs_labels_tensor):
            final_labels_tensor = eeg_labels_tensor
        else:
            print("警告：EEG和MEG(NIRS)的标签不一致。将使用EEG标签。")
            final_labels_tensor = eeg_labels_tensor
    elif eeg_labels_tensor is not None:
        final_labels_tensor = eeg_labels_tensor
    elif nirs_labels_tensor is not None:
        final_labels_tensor = nirs_labels_tensor
    else:
        print("错误：无法加载任何标签数据。请检查文件路径和内容。")

    return eeg_data_tensor, nirs_data_tensor, final_labels_tensor


# 在模块级别调用函数，并将结果赋给全局变量
# 这样，当其他脚本导入 data_read 时，这些变量就会被加载并可用
# eeg, nirs, labels = load_eeg_and_meg_data()
