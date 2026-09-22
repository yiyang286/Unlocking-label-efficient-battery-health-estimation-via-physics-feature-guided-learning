import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os
import gc
from tqdm import tqdm
from model.model import MTLEIS
from DSG_mtl_LOCV import FewShotDSGenerator_ealy_cycle_LOCV
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== 数据加载与归一化 =====================
class EIS_SemiSupervised_Dataset(Dataset):
    def __init__(self, eis_arr, labels=None, feat_labels=None):
        self.eis_arr = torch.FloatTensor(eis_arr)
        if labels is None:
            self.labels = torch.FloatTensor([-1.0] * len(eis_arr))
        else:
            self.labels = torch.FloatTensor(labels)
        if feat_labels is None:
            raise ValueError("feat_labels must be provided")
        self.feat_labels = torch.FloatTensor(feat_labels)

    def __len__(self):
        return len(self.eis_arr)

    def __getitem__(self, idx):
        return self.eis_arr[idx], self.labels[idx], self.feat_labels[idx]


def get_eis_dataloaders(dataset, exp_name="EXP3", batch_size=32, num_workers=0):
    exp_data = dataset[exp_name]
    train_labeled_eis = exp_data['train']['labeled_pairs']['images']
    train_unlabeled_eis = exp_data['train']['unlabeled_pairs']['images']
    train_all_eis = np.concatenate([train_labeled_eis, train_unlabeled_eis], axis=0)
    print('train_labeled_eis.shape', train_labeled_eis.shape)
    test_eis = exp_data['test']['labeled_pairs']['images']
    test_labels = exp_data['test']['labeled_pairs']['labels']

    # 获取特征标签
    train_labeled_feats = np.column_stack([
        exp_data['train']['labeled_pairs']['features']['f1'],
        exp_data['train']['labeled_pairs']['features']['f2'],
        exp_data['train']['labeled_pairs']['features']['f3']
    ])
    train_unlabeled_feats = np.column_stack([
        exp_data['train']['unlabeled_pairs']['features']['f1'],
        exp_data['train']['unlabeled_pairs']['features']['f2'],
        exp_data['train']['unlabeled_pairs']['features']['f3']
    ])
    test_feats = np.column_stack([
        exp_data['test']['labeled_pairs']['features']['f1'],
        exp_data['test']['labeled_pairs']['features']['f2'],
        exp_data['test']['labeled_pairs']['features']['f3']
    ])

    all_train_feats = np.concatenate([train_labeled_feats, train_unlabeled_feats], axis=0)
    scaler_feats = {'f1': MinMaxScaler(feature_range=(-1, 1)), 'f2': MinMaxScaler((-1, 1)), 'f3': MinMaxScaler((-1, 1))}
    train_f1 = all_train_feats[:, 0]
    train_f2 = all_train_feats[:, 1]
    train_f3 = all_train_feats[:, 2]
    scaler_feats['f1'].fit(train_f1.reshape(-1, 1))
    scaler_feats['f2'].fit(train_f2.reshape(-1, 1))
    scaler_feats['f3'].fit(train_f3.reshape(-1, 1))
    def normalize_feats(feats, scaler_feats):
        feats_norm = feats.copy()
        feats_norm[:, 0] = scaler_feats['f1'].transform(feats[:, 0].reshape(-1, 1)).reshape(-1)
        feats_norm[:, 1] = scaler_feats['f2'].transform(feats[:, 1].reshape(-1, 1)).reshape(-1)
        feats_norm[:, 2] = scaler_feats['f3'].transform(feats[:, 2].reshape(-1, 1)).reshape(-1)
        return feats_norm
    train_all_feats_norm = normalize_feats(all_train_feats, scaler_feats)
    test_feats_norm = normalize_feats(test_feats, scaler_feats)

    # 实部/虚部分别归一化
    scalers = {"real": MinMaxScaler((-1, 1)), "imag": MinMaxScaler((-1, 1)), 'freq': MinMaxScaler((-1, 1))}
    train_freq = train_all_eis[:, 0, :, :].reshape(-1, 1)
    train_real = train_all_eis[:, 1, :, :].reshape(-1, 1)
    train_imag = train_all_eis[:, 2, :, :].reshape(-1, 1)
    scalers['freq'].fit(train_freq)
    scalers["real"].fit(train_real)
    scalers["imag"].fit(train_imag)

    def normalize_eis(eis_arr, scalers):
        eis_norm = eis_arr.copy()
        eis_norm[:, 0, :, :] = scalers['freq'].transform(eis_arr[:, 0, :, :].reshape(-1, 1)).reshape(-1, 8, 8)
        eis_norm[:, 1, :, :] = scalers["real"].transform(eis_arr[:, 1, :, :].reshape(-1, 1)).reshape(-1, 8, 8)
        eis_norm[:, 2, :, :] = scalers["imag"].transform(eis_arr[:, 2, :, :].reshape(-1, 1)).reshape(-1, 8, 8)
        return eis_norm

    train_labeled_eis_norm = normalize_eis(train_labeled_eis, scalers)
    train_unlabeled_eis_norm = normalize_eis(train_unlabeled_eis, scalers)
    test_eis_norm = normalize_eis(test_eis, scalers)

    # 合并训练集
    train_all_eis_norm = np.concatenate([train_labeled_eis_norm, train_unlabeled_eis_norm], axis=0)
    train_all_labels = np.concatenate([exp_data['train']['labeled_pairs']['labels'], [-1.0] * len(train_unlabeled_eis_norm)],
                                      axis=0)

    train_dataset = EIS_SemiSupervised_Dataset(train_all_eis_norm, train_all_labels, train_all_feats_norm)
    test_dataset = EIS_SemiSupervised_Dataset(test_eis_norm, test_labels, test_feats_norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, scalers


# ===================== 训练与评估函数 =====================
def train_model(model, train_loader, optimizer, device, epochs=300):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for eis_batch, label_batch, feat_batch in train_loader:
            eis_batch = eis_batch.to(device)
            label_batch = label_batch.to(device)
            feat_batch = feat_batch.to(device)
            labeled_mask = (label_batch != -1.0)

            optimizer.zero_grad()
            _, _, _, loss = model(eis_batch, soh_labels=label_batch, labeled_mask=labeled_mask, feat_labels=feat_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for eis_batch, label_batch, _ in test_loader:
            eis_batch = eis_batch.to(device)
            soh_pred, _, _, _ = model(eis_batch)
            y_true_all.extend(label_batch.cpu().numpy())
            y_pred_all.extend(soh_pred.cpu().numpy())
    return np.array(y_true_all), np.array(y_pred_all)


# ===================== 主实验循环 =====================
if __name__ == '__main__':
    # ------------------- 实验配置 -------------------
    set_seed(42)

    # 实验参数
    num_labeled_cells_list = np.arange(6, 11, 2)
    num_repeats = 20  
    exp_name = "LOCV1"  # 选择实验组,该实验组是45度工况下的电池训练，35和23度工况下的电池用于测试
    batch_size = 32
    epochs = 500
    lr = 1e-3
    lambda_unlabel = 0.5
    latent_dim = 128
    project_root = r'C:\Users\Wenyanxiaoyao\Desktop\Unlocking-label-efficient-battery-health-estimation-via-physics-feature-guided-learning/'
    save_base_root = project_root + r'result/result_for_loco1/'
    save_path = save_base_root + "semi_MTL1_locv1.joblib"  # 结果保存路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ------------------- 初始化结果字典 -------------------
    all_results = {}
    for n in num_labeled_cells_list:
        all_results[f"num_cell_{n}"] = []

    # ------------------- 开始实验循环 -------------------
    for num_labeled_cell in num_labeled_cells_list:
        print(f"\n{'=' * 60}")
        print(f"开始标签数量 = {num_labeled_cell} 的实验 (共 {num_repeats} 次重复)")
        print(f"{'=' * 60}")
        save_path_temp = save_base_root + f"semi_MTL1_locv_up_to_{num_labeled_cell}cell.joblib"  # 结果保存路径

        for repeat_idx in tqdm(range(num_repeats), desc=f"标签电池数 {num_labeled_cell} 进度"):
            # 1. 设置随机种子（确保每次采样不同）
            current_seed = 1000 + repeat_idx  # 用不同的seed保证采样差异

            # 2. 生成数据集
            generator = FewShotDSGenerator_ealy_cycle_LOCV(random_seed=current_seed)
            raw_dataset = generator.generate_dataset(num_labeled_cells=num_labeled_cell)

            # 3. 生成DataLoader
            train_loader, test_loader, _ = get_eis_dataloaders(
                raw_dataset, exp_name=exp_name, batch_size=batch_size, num_workers=0
            )

            # 4. 初始化模型、优化器
            model = MTLEIS(in_channels=3, latent_dim=latent_dim, lambda_unlabel=lambda_unlabel).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # 5. 训练模型
            model = train_model(model, train_loader, optimizer, device, epochs=epochs)

            # 6. 评估模型
            y_true, y_pred = evaluate_model(model, test_loader, device)

            # 7. 保存本次实验结果
            all_results[f"num_cell_{num_labeled_cell}"].append({
                "seed": current_seed,
                "num_labeled_cell": num_labeled_cell,
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist()
            })
            print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
            print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}")

            # 8. 清理内存（防止内存溢出）
            del model, optimizer, train_loader, test_loader, raw_dataset, generator
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # 每完成一个标签数量设置下的重复实验，保存一次结果
        joblib.dump(all_results, save_path_temp)
        print(f"\n标签数 {num_labeled_cell} 完成，中间结果已保存到 {save_path_temp}")

    # ------------------- 最终保存 -------------------
    joblib.dump(all_results, save_path)
    print(f"\n{'=' * 60}")
    print(f"所有实验完成！最终结果已保存到: {save_path}")
    print(f"{'=' * 60}")
