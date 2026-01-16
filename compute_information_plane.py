
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset

NUM_INTERVALS = 50
NUM_LABEL = 10


# ---------------------- Discretization ----------------------
def Discretize_v2(layer_T):
    labels = np.arange(NUM_INTERVALS)
    bins = np.arange(NUM_INTERVALS + 1) / float(NUM_INTERVALS)

    for i in range(layer_T.shape[1]):
        temp = pd.cut(layer_T[:, i], bins, labels=labels)
        layer_T[:, i] = np.array(temp)
    return layer_T


# ---------------------- Mutual Information ----------------------
def MI_cal_v2(label_matrix, layer_T, NUM_TEST_MASK):

    MI_XT = 0
    MI_TY = 0

    # Softmax normalize
    layer_T = np.exp(layer_T - np.max(layer_T, axis=1, keepdims=True))
    layer_T /= np.sum(layer_T, axis=1, keepdims=True)

    # discretize
    layer_T = Discretize_v2(layer_T)

    # ------------------ I(X;T) ------------------
    XT_matrix = np.zeros((NUM_TEST_MASK, NUM_TEST_MASK))
    Non_repeat = []
    mark_list = []

    for i in range(NUM_TEST_MASK):
        pre_size = len(mark_list)

        if i == 0:
            Non_repeat.append(i)
            mark_list.append(i)
            XT_matrix[i, i] = 1

        else:
            for j in range(len(Non_repeat)):
                if (layer_T[i] == layer_T[Non_repeat[j]]).all():
                    mark_list.append(Non_repeat[j])
                    XT_matrix[i, Non_repeat[j]] = 1
                    break

        if pre_size == len(mark_list):
            Non_repeat.append(Non_repeat[-1] + 1)
            mark_list.append(Non_repeat[-1])
            XT_matrix[i, Non_repeat[-1]] = 1

    XT_matrix = np.delete(XT_matrix, range(len(Non_repeat), NUM_TEST_MASK), axis=1)
    XT_matrix = XT_matrix / NUM_TEST_MASK

    P_X = np.sum(XT_matrix, axis=1)
    P_T = np.sum(XT_matrix, axis=0)

    for i in range(XT_matrix.shape[0]):
        for j in range(XT_matrix.shape[1]):
            if XT_matrix[i, j] == 0:
                continue
            MI_XT += XT_matrix[i, j] * np.log2(XT_matrix[i, j] / (P_X[i] * P_T[j]))

    # ------------------ I(T;Y) ------------------
    TY_matrix = np.zeros((len(Non_repeat), NUM_LABEL))
    mark_list = np.array(mark_list)

    for i in range(len(Non_repeat)):
        TY_matrix[i, :] = np.sum(label_matrix[np.where(mark_list == i)[0], :], axis=0)

    TY_matrix = TY_matrix / NUM_TEST_MASK

    P_T2 = np.sum(TY_matrix, axis=1)
    P_Y2 = np.sum(TY_matrix, axis=0)

    for i in range(TY_matrix.shape[0]):
        for j in range(TY_matrix.shape[1]):
            if TY_matrix[i, j] == 0:
                continue
            MI_TY += TY_matrix[i, j] * np.log2(TY_matrix[i, j] / (P_T2[i] * P_Y2[j]))

    return MI_XT, MI_TY


# ---------------------- Batch MI Calculation ----------------------
def compute_batch_MI(target_model, dataset, batch_size=2000, device='cuda'):
    """
    Randomly sample batch_size samples from dataset,
    compute MI(X;T) and MI(T;Y) for that batch.
    Return one point: (I_XT, I_TY)
    """

    # ---- 修复 ----
    N = len(dataset)
    batch_size = min(batch_size, N)

    indices = np.random.choice(N, batch_size, replace=False)

    xs = []
    ys = []
    for idx in indices:
        x, y = dataset[idx]
        if x.dim() == 4:
            x = x.squeeze(0)
        xs.append(x.unsqueeze(0))
        ys.append(y)

    xs = torch.cat(xs, dim=0).to(device)
    ys = torch.tensor(ys).long().to(device)

    target_model.eval()
    with torch.no_grad():
        logits = target_model(xs).cpu().numpy()

    labels_onehot = F.one_hot(ys.cpu(), 10).numpy()

    IXT, ITY = MI_cal_v2(labels_onehot, logits, batch_size)
    return IXT, ITY


# ---------------- Training Dataset: Single Point ----------------
def compute_MI_full_dataset(target_model, dataset, device='cuda'):
    """
    使用整个 dataset（不采样、不分 batch）来计算 I(X;T) 和 I(T;Y)
    返回一个点 (IXT, ITY)
    """

    target_model.eval()

    X_all = []
    Y_all = []

    # -------- 取全量数据 --------
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_all.append(x.unsqueeze(0))
        Y_all.append(y)

    X_all = torch.cat(X_all, dim=0).to(device)         # shape [N, 3, 32, 32]
    Y_all = torch.tensor(Y_all).long().to(device)      # shape [N]

    # -------- 前向得到 logits --------
    with torch.no_grad():
        logits = target_model(X_all).cpu().numpy()

    # -------- 转 one-hot --------
    labels_onehot = F.one_hot(Y_all.cpu(), 10).numpy()

    # -------- 调用你的 MI 计算 --------
    IXT, ITY = MI_cal_v2(labels_onehot, logits, NUM_TEST_MASK=len(dataset))

    return IXT, ITY


# ---------------- Generated Dataset: Multiple Points ------------
'''def compute_dataset_MI_balanced(target_model, dataset, device='cuda',
                                samples_per_class=2000, num_points=50, num_classes=10):
    """
    计算 MI，但每次采样保证每一类都有 samples_per_class 个样本 (例如 200)。
    """
    print(f"⚖️ Computing MI with balanced sampling: {samples_per_class} samples/class...")

    XT_list = []
    TY_list = []

    # 1. 预处理：将数据集按类别分类索引
    # 假设 dataset 是 TensorDataset (imgs, labels)
    if isinstance(dataset, TensorDataset):
        all_labels = dataset.tensors[1].cpu().numpy()
    else:
        # 如果是普通 Dataset，需要遍历一次取出所有标签 (较慢但通用)
        all_labels = np.array([y for _, y in dataset])

    class_indices = {i: np.where(all_labels == i)[0] for i in range(num_classes)}

    # 检查是否有类别的样本不足 samples_per_class
    for i in range(num_classes):
        if len(class_indices[i]) < samples_per_class:
            print(f"⚠️ Warning: Class {i} only has {len(class_indices[i])} samples, but requested {samples_per_class}.")

    # 2. 循环多次进行采样和计算
    for i in range(num_points):
        selected_indices = []

        # 对每个类别抽取指定数量的索引
        for cls in range(num_classes):
            indices = class_indices[cls]
            # 如果样本够，无放回抽样；如果不够，有放回抽样(replace=True)
            replace = len(indices) < samples_per_class
            sampled_idx = np.random.choice(indices, samples_per_class, replace=replace)
            selected_indices.extend(sampled_idx)

        # 3. 构建临时子数据集
        # Subset 接收原始 dataset 和 索引列表
        subset_ds = Subset(dataset, selected_indices)

        # 4. 计算该均衡子集的 MI
        # 注意：这里我们调用 compute_MI_full_dataset，因为它能处理这 2000 个样本的整体分布
        IXT, ITY = compute_MI_full_dataset(target_model, subset_ds, device=device)

        # 结果转为 float (如果是 tensor)
        if isinstance(IXT, torch.Tensor): IXT = IXT.item()
        if isinstance(ITY, torch.Tensor): ITY = ITY.item()

        XT_list.append(IXT)
        TY_list.append(ITY)

        if (i + 1) % 10 == 0:
            print(f"   Sampled batch {i + 1}/{num_points}")

    return XT_list, TY_list'''


'''def compute_dataset_MI_balanced(target_model, dataset, device='cuda',
                                samples_per_class=100,  # [修改] 每类取 500 张
                                num_points=50,  # [修改] 总共计算 50 次 (得到 50 个点)
                                num_classes=10):
    """
    计算 MI:
    1. 循环 num_points 次 (50次)。
    2. 每次循环中，从 dataset 里为每个类别随机抽取 samples_per_class (500) 张图片。
    3. 组成一个临时的 balanced subset (5000张)。
    4. 计算一次 MI，存入列表。
    """
    print(f"⚖️ Computing MI with balanced sampling: {samples_per_class} samples/class, {num_points} points...")

    XT_list = []
    TY_list = []

    # 1. 预处理：将数据集按类别分类索引
    # 假设 dataset 是 TensorDataset (imgs, labels)
    if isinstance(dataset, TensorDataset):
        all_labels = dataset.tensors[1].cpu().numpy()
    else:
        # 如果是普通 Dataset，需要遍历一次取出所有标签
        all_labels = np.array([y for _, y in dataset])

    class_indices = {i: np.where(all_labels == i)[0] for i in range(num_classes)}

    # 检查是否有类别的样本不足 samples_per_class
    for i in range(num_classes):
        curr_len = len(class_indices[i])
        if curr_len < samples_per_class:
            print(
                f"⚠️ Warning: Class {i} only has {curr_len} samples, but requested {samples_per_class}. (Will sample with replacement)")

    # 2. 循环 num_points 次 (50次)
    for i in range(num_points):
        selected_indices = []

        # 对每个类别抽取指定数量的索引 (500个)
        for cls in range(num_classes):
            indices = class_indices[cls]
            # 如果样本够，无放回抽样(False)；如果不够，有放回抽样(True)
            replace = len(indices) < samples_per_class

            sampled_idx = np.random.choice(indices, samples_per_class, replace=replace)
            selected_indices.extend(sampled_idx)

        # 3. 构建临时子数据集 (总共 500 * 10 = 5000 张)
        # Subset 接收原始 dataset 和 索引列表
        subset_ds = Subset(dataset, selected_indices)

        # 4. 计算该均衡子集的 MI (得到 1 个点)
        # 调用 compute_MI_full_dataset 处理这 5000 个样本
        IXT, ITY = compute_MI_full_dataset(target_model, subset_ds, device=device)

        # 结果转为 float (如果是 tensor)
        if isinstance(IXT, torch.Tensor): IXT = IXT.item()
        if isinstance(ITY, torch.Tensor): ITY = ITY.item()
        print(IXT,ITY)

        XT_list.append(IXT)
        TY_list.append(ITY)

        if (i + 1) % 10 == 0:
            print(f"   Sampled batch {i + 1}/{num_points} -> IXT: {IXT:.4f}, ITY: {ITY:.4f}")

    return XT_list, TY_list'''


def compute_dataset_MI_multiple_batches(target_model, dataset, device='cuda',
                                        batch_size=40000, num_points=50):
    XT_list = []
    TY_list = []

    for _ in range(num_points):
        IXT, ITY = compute_batch_MI(target_model, dataset, batch_size, device)
        XT_list.append(IXT)
        TY_list.append(ITY)

    return XT_list, TY_list


def compute_dataset_MI_balanced_with_MIcal(
    target_model,
    dataset,
    device='cuda',
    samples_per_class=800,   # 每个类采 200 个，总共 200*10 = 2000
    num_classes=10,
    num_points=50,           # 采样 50 次 → 50 个 (IXT, ITY)
    batch_size=256
):
    """
    返回:
        IXT_list: [num_points]，每次采样得到的 I(X;T)
        ITY_list: [num_points]，每次采样得到的 I(T;Y)
    """
    target_model.eval()

    # 1. 先把整个 dataset 的 label 抽出来，按类别存索引
    if isinstance(dataset, TensorDataset):
        all_labels = dataset.tensors[1].cpu().numpy()
    else:
        # 通用 Dataset，稍微慢一点遍历
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

    class_indices = {c: np.where(all_labels == c)[0] for c in range(num_classes)}

    # 检查每一类是否够 samples_per_class
    for c in range(num_classes):
        if len(class_indices[c]) < samples_per_class:
            raise ValueError(
                f"类别 {c} 样本不足: 只有 {len(class_indices[c])} 个, "
                f"但需要 samples_per_class={samples_per_class}"
            )

    IXT_list = []
    ITY_list = []

    for it in range(num_points):
        print(f"🔁 Sampling round {it+1}/{num_points} ...")

        # 2. 为每一类随机采样 samples_per_class 个索引
        chosen_indices = []
        for c in range(num_classes):
            idx_c = class_indices[c]
            chosen_c = np.random.choice(idx_c, size=samples_per_class, replace=False)
            chosen_indices.append(chosen_c)
        chosen_indices = np.concatenate(chosen_indices)
        np.random.shuffle(chosen_indices)

        # 总样本数（相当于你的 mask = NUM_TEST_MASK）
        NUM_TEST_MASK = chosen_indices.shape[0]   # ~ 2000

        # 3. 建一个子集 DataLoader，跑一遍 target_model 拿 logits 和 labels
        subset = Subset(dataset, chosen_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        all_logits = []
        all_y = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                out = target_model(x_batch)
                # 兼容 (logits, feat) 和 logits 两种情况
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(y_batch.detach().cpu().numpy())

        layer_T = np.concatenate(all_logits, axis=0)   # [N, num_classes]
        y_np = np.concatenate(all_y, axis=0)           # [N]

        assert layer_T.shape[0] == NUM_TEST_MASK

        # 4. 构造 one-hot label_matrix: [N, NUM_LABEL]
        label_matrix = np.zeros((NUM_TEST_MASK, NUM_LABEL), dtype=np.float32)
        label_matrix[np.arange(NUM_TEST_MASK), y_np] = 1.0

        # 5. 用你给的 MI_cal_v2 计算 MI
        MI_XT, MI_TY = MI_cal_v2(label_matrix, layer_T, NUM_TEST_MASK)
        print(f"   -> IXT = {MI_XT:.4f}, ITY = {MI_TY:.4f}")

        IXT_list.append(MI_XT)
        ITY_list.append(MI_TY)

    return np.array(IXT_list), np.array(ITY_list)


