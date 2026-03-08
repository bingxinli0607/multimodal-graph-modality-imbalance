# ==============================================================================
# 文件：train_imbalance.py
# 功能：在 train_baseline.py 的基础上，模拟"模态不平衡"场景，研究模型的鲁棒性
#
# 与 baseline 的核心区别（这才是这个文件存在的意义）：
#   baseline 假设两个视图（引文图 + kNN图）永远完整可用——这是理想情况。
#   现实中，数据经常残缺：
#     比如：论文的特征向量（视图B依赖的原料）在某些节点上缺失、损坏、或根本没有。
#   本文件用两种手段模拟这种"视图B残缺"的场景，测试模型是否还能撑住：
#     1. miss_b   ：评估时随机丢掉视图B的一部分边（模拟测试时模态残缺）
#     2. moddrop  ：训练时以一定概率把视图B整体清空（逼迫模型学会只靠视图A也能工作）
#
# 注意：本文件与 train_baseline.py 完全独立，没有 import 关系，
#       DualGCN / eval_acc 等都重新定义了一遍（代码复用性可以后续改进）
# ==============================================================================

# ── 导入工具包 ──────────────────────────────────────────────────────────────────
import argparse                              # 解析命令行参数
import random                                # Python 内置随机模块（用于固定随机种子）
import torch                                 # PyTorch 深度学习库
import torch.nn.functional as F             # relu / dropout / cross_entropy 等函数
from torch import nn                         # 神经网络模块
from torch_geometric.datasets import Planetoid  # Cora 等图数据集
from torch_geometric.nn import GCNConv          # 图卷积层
from torch_geometric.nn.pool import knn_graph   # k近邻图构建


# ==============================================================================
# 工具函数1：固定所有随机种子
# ==============================================================================
def set_seed(seed: int):
    """
    同时固定 Python 和 PyTorch 的随机种子，确保实验可复现。

    为什么要固定两个？
      - Python 的 random 模块和 PyTorch 的随机数生成器是两套独立的系统
      - 只固定一个，另一个仍然随机，结果还是会变
      - 这里没有固定 numpy，如果后续用到 numpy 随机操作还需加 np.random.seed(seed)
    """
    random.seed(seed)        # 固定 Python 内置 random 模块的种子
    torch.manual_seed(seed)  # 固定 PyTorch 的种子（包括 CPU 和 GPU）


# ==============================================================================
# 工具函数2：随机丢边（模拟视图B的边缺失）
# ==============================================================================
def drop_edge(edge_index: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    从边集合中随机删除一部分边，用来模拟"模态残缺"的现实场景。

    打个比方：
      你有100条朋友关系（边），突然有些人删了账号（数据丢失），
      只剩下 (1 - drop_prob) 比例的关系还存在。

    参数：
      edge_index : 原始边索引，形状 [2, 边数]
      drop_prob  : 丢边概率，取值 [0, 1]。0=不丢，0.5=随机丢一半，1=全丢

    返回：
      保留下来的边索引，形状 [2, 剩余边数]
    """
    # 如果不丢边，直接原样返回，不做任何操作（优化：避免无效计算）
    if drop_prob <= 0:
        return edge_index

    E = edge_index.size(1)          # 取出总边数（size(1) = 第1维的长度 = 列数）

    # torch.rand(E) 生成 E 个 [0,1) 之间的随机数
    # > drop_prob 得到布尔数组：随机数大于阈值的位置为 True（保留），否则为 False（丢弃）
    # 例如 drop_prob=0.5：大约一半的位置是 True，一半是 False
    keep = torch.rand(E) > drop_prob

    # edge_index[:, keep]：用布尔数组作为列选择器，只保留 keep=True 的列（边）
    return edge_index[:, keep]


# ==============================================================================
# 核心模型：DualGCN（与 baseline 完全相同，此处重新定义）
#
# 结构说明请参考 train_baseline.py 中的详细注释，这里不再重复。
# 两层 GCN × 两个视图 → 平均融合 → 输出分类 logits
# ==============================================================================
class DualGCN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        super().__init__()
        # 视图A的两层GCN（处理引文图）
        self.gcn1_a = GCNConv(in_dim, hid)
        self.gcn2_a = GCNConv(hid, out_dim)

        # 视图B的两层GCN（处理kNN特征图）
        self.gcn1_b = GCNConv(in_dim, hid)
        self.gcn2_b = GCNConv(hid, out_dim)

        self.dropout = dropout

    def forward(self, x, edge_a, edge_b):
        # ── 视图A：通过引文图聚合邻居信息 ────────────────────────────────────
        xa = F.relu(self.gcn1_a(x, edge_a))
        xa = F.dropout(xa, p=self.dropout, training=self.training)
        la = self.gcn2_a(xa, edge_a)   # 形状 [2708, 7]

        # ── 视图B：通过kNN图聚合邻居信息 ──────────────────────────────────────
        # 注意：当 edge_b 为空（0条边）时，GCNConv 会退化为只用节点自身特征
        # 这正是"视图B完全缺失"时模型的行为——它仍能运行，只是少了来自邻居的信息
        xb = F.relu(self.gcn1_b(x, edge_b))
        xb = F.dropout(xb, p=self.dropout, training=self.training)
        lb = self.gcn2_b(xb, edge_b)   # 形状 [2708, 7]

        # ── 两视图结果平均融合 ─────────────────────────────────────────────────
        logits = 0.5 * la + 0.5 * lb   # 形状 [2708, 7]
        return logits, la, lb


# ==============================================================================
# 评估函数：计算训练/验证/测试集准确率（与 baseline 相同）
# ==============================================================================
@torch.no_grad()  # 评估时关闭梯度计算，节省内存和时间
def eval_acc(model, data, edge_a, edge_b):
    model.eval()  # 切换评估模式，关闭 Dropout
    out, _, _ = model(data.x, edge_a, edge_b)
    pred = out.argmax(dim=-1)  # 取得分最高的类别作为预测结果
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs  # [训练准确率, 验证准确率, 测试准确率]


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    # ── 第一步：解析命令行参数 ────────────────────────────────────────────────
    p = argparse.ArgumentParser()
    # 以下参数与 baseline 相同
    p.add_argument("--k",      type=int,   default=10)
    p.add_argument("--epochs", type=int,   default=200)
    p.add_argument("--hid",    type=int,   default=64)
    p.add_argument("--lr",     type=float, default=0.01)
    p.add_argument("--wd",     type=float, default=5e-4)
    p.add_argument("--seed",   type=int,   default=7)

    # ── 新增！模态不平衡相关参数 ──────────────────────────────────────────────
    # miss_b：评估时视图B的丢边概率
    #   0.0 = 完整（与baseline相同）
    #   0.5 = 评估时随机丢掉50%的kNN边
    #   1.0 = 评估时视图B完全消失
    # 用途：测试"部署到真实场景时，特征模态有损坏"模型还能有多准
    p.add_argument("--miss_b",  type=float, default=0.0,
                   help="edge drop prob for view-B at eval time")

    # moddrop：训练时"整个视图B消失"的概率
    #   0.0 = 训练时视图B始终完整（与baseline相同）
    #   0.5 = 训练时每个epoch有50%概率把视图B整体清空
    # 用途：让模型在训练时就见过"没有视图B"的情况，提前练习只靠视图A分类
    #       这样测试时即使视图B残缺，模型也不会"懵"
    p.add_argument("--moddrop", type=float, default=0.0,
                   help="train-time modality dropout prob for view-B")

    args = p.parse_args()
    set_seed(args.seed)

    # ── 第二步：加载数据，构建两个视图的图 ────────────────────────────────────
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]

    edge_a = data.edge_index                          # 视图A：原始引文图（固定不变）
    edge_b_full = knn_graph(data.x, k=args.k, loop=False)  # 视图B：完整kNN图（后续会按需削减）

    # ── 第三步：初始化模型和优化器 ────────────────────────────────────────────
    model = DualGCN(dataset.num_features, args.hid, dataset.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val  = 0.0
    best_test = 0.0

    # ── 第四步：在训练循环开始前，生成一次固定的评估用残缺图 ──────────────────
    # 为什么只生成一次？
    #   drop_edge 每次调用都会重新随机丢边，丢掉的具体边每次都不同。
    #   如果放在循环内部，每个 epoch 评估时面对的残缺图都不一样，
    #   val_acc 的变化就混入了"随机丢边运气"的噪声，无法判断是模型真的变好了。
    #   固定下来后，200轮用同一把尺子衡量，best_val 的比较才有意义。
    edge_b_eval = drop_edge(edge_b_full, args.miss_b)

    # ── 第五步：训练循环 ───────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        # ── 关键创新点1：训练时模态随机丢弃（Modality Dropout）─────────────────
        # torch.rand(1).item() 生成一个 [0,1) 的随机数
        # 如果这个随机数 < moddrop，就把视图B整体清空（模拟"这轮训练没有特征模态"）
        # 例如 moddrop=0.5：每个epoch各有50%的概率完整/缺失视图B
        if args.moddrop > 0 and torch.rand(1).item() < args.moddrop:
            # edge_b_full[:, :0] 是一个"0列"的空矩阵，形状 [2, 0]，即0条边
            # 这不是报错，而是合法的"空图"——相当于视图B一条边都没有
            edge_b_train = edge_b_full[:, :0]  # 视图B完全消失
        else:
            edge_b_train = edge_b_full          # 视图B正常使用

        # 用（可能残缺的）edge_b_train 做前向传播和损失计算
        out, _, _ = model(data.x, edge_a, edge_b_train)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        # ── 关键创新点2：用固定的残缺图评估 ───────────────────────────────────
        # edge_b_eval 在循环外已经固定，每轮评估条件完全一致
        train_acc, val_acc, test_acc = eval_acc(model, data, edge_a, edge_b_eval)

        # 用验证集选最优时刻（见 baseline 注释中的解释）
        if val_acc > best_val:
            best_val  = val_acc
            best_test = test_acc

        # 每50轮打印一次（比 baseline 的20轮更稀疏，因为实验要跑很多次）
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val {val_acc:.4f} | best_test {best_test:.4f}")

    # 最终汇报：同时打印实验条件（miss_b / moddrop）和结果，方便对比多组实验
    print(f"Summary: miss_b={args.miss_b:.2f}, moddrop={args.moddrop:.2f}, best_test={best_test:.4f}")

if __name__ == "__main__":
    main()