import joblib
import os
import numpy as np
from utilities.eis_data_extractor import DataExtractorSpecific
from utilities.feature_calculator import extract_f1_features, extract_f2_f3_features, remove_nan_indices
import pandas as pd


def filter_invalid_labels(x1_list, x2_list, y_list):
    filtered = [(x1, x2, y) for x1, x2, y in zip(x1_list, x2_list, y_list) if
                not (y == -1 or y > 1e4 or y < 0.6)]  # 筛选有效的SOH范围
    if not filtered:
        return [], [], []
    filtered_x1, filtered_x2, filtered_y = zip(*filtered)
    return list(filtered_x1), list(filtered_x2), list(filtered_y)


class FewShotDSGenerator_ealy_cycle_LOCV:
    """
    少样本数据集生成器类
    通过设置num_few_shot参数，每次运行时随机选择不同样本，返回对应的数据集
    """

    def __init__(self, source_root=None, random_seed=None, soc_filter=100):
        """
        初始化数据集生成器

        Args:
            source_root: 项目根目录路径
            random_seed: 随机种子，为None时每次随机选择不同样本
        """
        self.source_root = source_root or r'your own project root'
        self.random_seed = random_seed
        self.soc_filter = soc_filter  # 筛选SOC值在100%的样本
        print(f"筛选SOC值在{self.soc_filter}%的样本")
        self._load_data()

    def _load_data(self):
        # 加载数据集
        self.dataset = joblib.load('processed_dataset1.joblib')
        self.DE = DataExtractorSpecific(self.dataset, soc_filter=self.soc_filter)

        # define the training and test cells
        condition1 = ['T45-1', 'T45-2', 'T45-3', 'T45-4', 'T45-5', 'T45-6', 'T45-7', 'T45-8', 'T45-9']
        condition2 = ['T35-1', 'T35-2', 'T35-3']
        condition3 = ['T23-1', 'T23-2', 'T23-3', 'T23-4']
        condition1_d = ['T45D70-1', 'T45D70-2', 'T45D50-1', 'T45D50-2', 'T45D50-3', 'T45D50-4', 'T45D30-1', 'T45D30-2']

        train_cells1 = condition1+condition1_d   # 45 degree
        test_cells1 = condition2+condition3   # 35 and 23 degree

        cell_id_mapping = pd.read_excel('cell_id_mapping.xlsx')  # mapping between old and new cids
        new_ids = cell_id_mapping['New Cell_ID'].values
        old_ids = cell_id_mapping['Original Cell_ID'].values
        id_mapping_dict = {new_ids[i]: old_ids[i] for i in range(len(old_ids))}

        # 新旧电池ID映射
        train_cells1 = [id_mapping_dict[cell_id] for cell_id in train_cells1]
        test_cells1 = [id_mapping_dict[cell_id] for cell_id in test_cells1]

        self.EXP_LIST = [
            [train_cells1, test_cells1]
        ]

    def generate_dataset(self, num_labeled_cells, save_path=None):
        """
        生成少样本数据集

        Args:
            num_labeled_cells: 训练集中选择的标签样本数量
            save_path: 保存路径，为None时不保存

        Returns:
            dict: 包含所有实验组数据的字典
        """
        # 设置随机种子（如果提供）
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        all_features = {}

        for i, experiment in enumerate(self.EXP_LIST):
            train_cell_id_list = experiment[0]
            test_cell_id_list = experiment[1]

            # 生成单个实验组的数据
            while True:
                exp_features = self._generate_experiment_data(i + 1, train_cell_id_list, test_cell_id_list,
                                                              num_labeled_cells)
                self.random_seed += 1000
                if exp_features['train'] is not None:
                    break

            all_features[f'LOCV{i + 1}'] = exp_features

        # 打印数据统计信息
        self._print_dataset_info(all_features, num_labeled_cells)

        # 保存数据
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(all_features, save_path)
            print(f"数据已保存到: {save_path}")

        return all_features

    def _generate_experiment_data(self, exp_num, train_cell_id_list, test_cell_id_list, num_labeled_cells):
        """首先随机选择指定数量的有标签电池"""
        import random
        random.seed(self.random_seed)
        if num_labeled_cells >= len(train_cell_id_list):
            print(
                f"num_labeled_cells {num_labeled_cells} 大于等于训练集电池数 {len(train_cell_id_list)}，所有电池作为标签样本")
            train_labeled_cells = train_cell_id_list
            train_unlabeled_cells = train_cell_id_list
        else:
            # cell level separate
            train_labeled_cells = random.sample(train_cell_id_list, num_labeled_cells)
            train_unlabeled_cells = train_cell_id_list

        def extract_data(cell_id_list, type='labeled'):
            """提取指定电池ID列表中的所有数据"""
            x_eis, x_labels, x_f1, x_f2, x_f3 = [], [], [], [], []
            for cell_id in cell_id_list:
                eis_list, eis_image_list, y_list = self.DE.extract_data([cell_id])  # 这里的eis_list中的元素是eis array
                # 获取meta data
                meta_dict = self.DE.get_meta_data()
                cycles = meta_dict['cycles']

                def filter_100_cycle_data(cycles, eis_list, eis_image_list, y_list):
                    """筛选出 cycle <= 100 的样本"""
                    valid_indices = [i for i, c in enumerate(cycles) if c <= 100]
                    valid_eis = [eis_list[i] for i in valid_indices]
                    valid_eis_image = [eis_image_list[i] for i in valid_indices]
                    valid_y = [y_list[i] for i in valid_indices]
                    return valid_eis, valid_eis_image, valid_y

                if type == 'labeled':
                    eis_list, eis_image_list, y_list = filter_100_cycle_data(cycles, eis_list, eis_image_list, y_list)

                # 去除掉soh为-1的样本，无效
                eis_list, eis_image_list, y_list = filter_invalid_labels(eis_list, eis_image_list, y_list)

                # Extract features
                f1 = extract_f1_features(eis_list)
                f2_f3_list = [extract_f2_f3_features(item[:, 0], item[:, 1], cell_id) for item in eis_list]
                if len(f2_f3_list) == 0:  # 当没有有效特征时，跳过该电池
                    continue
                f2, f3 = zip(*f2_f3_list)  # may contain NaN values

                # Remove NaN indices
                f1, f2, f3, y_list, eis_image_list = remove_nan_indices(f1, f2, f3, y_list, eis_image_list)

                x_eis.extend(eis_image_list)
                x_labels.extend(y_list)
                x_f1.extend(f1)
                x_f2.extend(f2)
                x_f3.extend(f3)
            return np.array(x_eis), np.array(x_labels), np.array(x_f1), np.array(x_f2), np.array(x_f3)

        all_train_labeled_eis, all_train_labeles, all_train_labeled_f1, all_train_labeled_f2, all_train_labeled_f3 = \
            extract_data(train_labeled_cells, type='labeled')
        all_train_unlabeled_eis, _, all_train_unlabeled_f1, all_train_unlabeled_f2, all_train_unlabeled_f3 = \
            extract_data(train_unlabeled_cells, type='unlabeled')

        # =================================检查标签eis列表是否为空=====================================
        if len(all_train_labeled_eis) == 0:
            print('-----------------------没有EIS标签数据， 重新采样---------------------------')
            return {'train': None, 'test': None}

        # 打印实际的标签样本数量
        print(f"LOCV{exp_num} 实际标签样本数量: {len(all_train_labeles)}")
        print(f"LOCV{exp_num} 标签数据形状: images={all_train_labeled_eis.shape}, labels={all_train_labeles.shape}")
        print(f"LOCV{exp_num} 无标签数据形状: images={all_train_unlabeled_eis.shape}")

        # 生成测试数据
        test_features = self._generate_test_data(test_cell_id_list)

        # 返回实验组数据
        return {
            'train': {
                'labeled_pairs': {
                    'images': all_train_labeled_eis.transpose(0, 3, 1, 2),
                    'labels': all_train_labeles,
                    'features': {
                        'f1': all_train_labeled_f1,
                        'f2': all_train_labeled_f2,
                        'f3': all_train_labeled_f3
                    }
                },
                'unlabeled_pairs': {
                    'images': all_train_unlabeled_eis.transpose(0, 3, 1, 2),
                    'features': {
                        'f1': all_train_unlabeled_f1,
                        'f2': all_train_unlabeled_f2,
                        'f3': all_train_unlabeled_f3
                    }
                }
            },
            'test': test_features
        }

    def _generate_test_data(self, test_cell_id_list):
        """生成测试数据"""
        test_images, test_labels, test_f1, test_f2, test_f3, test_temps = [], [], [], [], [], []

        for cell_id in test_cell_id_list:
            eis_list, eis_image_list, y_list = self.DE.extract_data([cell_id])
            eis_list, eis_image_list, y_list = filter_invalid_labels(eis_list, eis_image_list, y_list)

            # Extract features
            f1 = extract_f1_features(eis_list)
            f2_f3_list = [extract_f2_f3_features(item[:, 0], item[:, 1], cell_id) for item in eis_list]
            f2, f3 = zip(*f2_f3_list)  # may contain NaN values

            # Remove NaN indices
            f1, f2, f3, y_list, eis_image_list = remove_nan_indices(f1, f2, f3, y_list, eis_image_list)

            test_images.extend(eis_image_list)
            test_labels.extend(y_list)
            test_f1.extend(f1)
            test_f2.extend(f2)
            test_f3.extend(f3)

        return {
            'labeled_pairs': {
                'images': np.array(test_images).transpose(0, 3, 1, 2),
                'labels': np.array(test_labels),
                'features': {
                    'f1': np.array(test_f1),
                    'f2': np.array(test_f2),
                    'f3': np.array(test_f3)
                }
            }
        }

    def _print_dataset_info(self, all_features, num_labeled_cells):
        """打印数据集信息"""
        print(f"\n=== 数据集信息 (num_labeled_cells={num_labeled_cells}) ===")
        for exp_key in all_features.keys():
            print(f"\n{exp_key}:")
            exp_data = all_features[exp_key]
            print(f"  训练集标签样本数: {exp_data['train']['labeled_pairs']['labels'].shape[0]}")
            print(f"  训练集无标签样本数: {exp_data['train']['unlabeled_pairs']['images'].shape[0]}")
            print(f"  测试集样本数: {exp_data['test']['labeled_pairs']['labels'].shape[0]}")


# 使用示例
if __name__ == '__main__':
    # 创建数据集生成器
    generator = FewShotDSGenerator_ealy_cycle_LOCV(random_seed=1000, soc_filter=100)

    # 生成不同少样本数量的数据集
    for num_labeled_cells in [6]:
        print(f"\n{'=' * 50}")
        print(f"生成 {num_labeled_cells} 个有标签电池的数据集")
        print(f"{'=' * 50}")

        # 生成数据集
        dataset = generator.generate_dataset(num_labeled_cells=num_labeled_cells)
        print(dataset['LOCV1']['train']['labeled_pairs']['labels'].shape)
        print(dataset['LOCV1']['train']['labeled_pairs']['labels'])
