# ==============================================================================
# 文件：train_baseline.py
# 功能：用"双视图图卷积网络 (Dual-GCN)"在 Cora 引文数据集上做节点分类
#
# 背景知识（新手必读）：
#   - 图(Graph)：由"节点"和"边"组成的数据结构。
#     比如论文引用网络：每篇论文是一个节点，论文A引用了论文B就连一条边。
#   - 图神经网络(GNN)：能处理图结构数据的神经网络。
#     它的核心思想是"消息传递"——每个节点会收集邻居节点的信息来更新自己。
#   - GCN（图卷积网络）：GNN 的经典实现之一，就像 CNN 处理图片一样处理图数据。
#   - 节点分类任务：给图中的每个节点打标签（比如：这篇论文属于哪个领域？）
# ==============================================================================

# ── 导入工具包 ──────────────────────────────────────────────────────────────────
import argparse                              # 解析命令行参数，让你在终端运行时可以传入选项
import torch                                 # PyTorch：深度学习的核心库，提供张量(Tensor)运算和自动求导
import torch.nn.functional as F             # 常用的神经网络"函数"集合，比如 relu、dropout、cross_entropy
from torch import nn                         # nn 模块：包含神经网络的各种"积木"，如线性层、卷积层等
from torch_geometric.datasets import Planetoid  # PyG 提供的经典图数据集加载器（Cora/CiteSeer/PubMed）
from torch_geometric.nn import GCNConv          # GCN 卷积层：图卷积的核心操作单元
from torch_geometric.nn.pool import knn_graph   # k近邻图构建函数：根据特征相似度自动连边


# ==============================================================================
# 核心模型定义：DualGCN（双视图图卷积网络）
#
# 为什么叫"双视图"？
#   一个节点可以从两个不同的"角度"（视图）来理解：
#   视图A（View A）：利用原始的引文关系图，谁引用了谁就连边 → 反映论文的学术关联
#   视图B（View B）：根据论文的特征向量计算相似度，相似的论文连边 → 反映内容相似性
#   最后把两个视图的预测结果平均融合，希望取长补短、提升准确率
# ==============================================================================
class DualGCN(nn.Module):
    # nn.Module 是所有 PyTorch 神经网络模型的基类，必须继承它
    # 继承后就能使用 .parameters()、.train()、.eval() 等便捷方法

    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        """
        模型的"图纸"——定义模型有哪些层，但还不执行任何计算。

        参数说明：
          in_dim   : 输入特征的维度。Cora 每篇论文有 1433 个词袋特征，所以 in_dim=1433
          hid      : 隐藏层维度，即中间表示的大小，默认 64。可以理解为"压缩后的特征数"
          out_dim  : 输出维度 = 类别数量。Cora 有 7 个类（7 个研究领域），所以 out_dim=7
          dropout  : Dropout 比率，默认 0.5。训练时随机"关掉"50%的神经元，防止过拟合
        """
        super().__init__()  # 必须调用父类的初始化方法，固定写法

        # ── 视图 A 的两层 GCN（处理引文图） ─────────────────────────────────────
        # 第一层：将 1433 维特征压缩到 64 维（特征提取）
        self.gcn1_a = GCNConv(in_dim, hid)
        # 第二层：将 64 维进一步映射到 7 维（每维对应一个类别的得分）
        self.gcn2_a = GCNConv(hid, out_dim)

        # ── 视图 B 的两层 GCN（处理 kNN 特征图） ────────────────────────────────
        # 结构和视图A完全相同，但是两套独立的参数，分别从不同的图结构中学习
        self.gcn1_b = GCNConv(in_dim, hid)
        self.gcn2_b = GCNConv(hid, out_dim)

        # 保存 dropout 比率，forward 里会用到
        self.dropout = dropout

    def forward(self, x, edge_a, edge_b):
        """
        模型的"流水线"——定义数据如何流过各层，执行实际的前向计算。

        参数说明：
          x      : 节点特征矩阵，形状 [节点数, 特征维度]，即 [2708, 1433]
                   可以理解为：2708 篇论文，每篇用 1433 个数字描述
          edge_a : 引文图的边索引，形状 [2, 边数]
                   第0行存起点，第1行存终点，表示哪些节点之间有引用关系
          edge_b : kNN图的边索引，同上，但边是根据特征相似度计算的
        """

        # ── 视图 A：通过引文图传播信息 ────────────────────────────────────────
        # 第一层 GCN：聚合邻居信息后，用 ReLU 激活（把负数截断为0，引入非线性）
        # xa 形状：[2708, 64]，每个节点得到一个 64 维的"中间表示"
        xa = F.relu(self.gcn1_a(x, edge_a))

        # Dropout：训练时随机丢弃 50% 的神经元输出（置为0），防止模型死记硬背
        # training=self.training 表示：只在训练模式下生效，评估时自动关闭
        xa = F.dropout(xa, p=self.dropout, training=self.training)

        # 第二层 GCN：从 64 维映射到 7 维，得到每个节点属于各类别的原始得分（logits）
        # la 形状：[2708, 7]
        la = self.gcn2_a(xa, edge_a)

        # ── 视图 B：通过 kNN 特征图传播信息 ──────────────────────────────────
        # 流程和视图A完全一样，只是换了一套参数和不同的图结构（edge_b）
        xb = F.relu(self.gcn1_b(x, edge_b))
        xb = F.dropout(xb, p=self.dropout, training=self.training)
        # lb 形状：[2708, 7]
        lb = self.gcn2_b(xb, edge_b)

        # ── 朴素融合（Naive Fusion）：两个视图的预测结果各取一半，直接平均 ───────
        # 这是最简单的多视图融合方式，后续研究会探索更智能的融合策略
        # logits 形状：[2708, 7]，数值最大的那一维就是预测的类别
        logits = 0.5 * la + 0.5 * lb

        # 返回三个值：融合结果、视图A的单独结果、视图B的单独结果
        # 后两个值目前没用到，但保留着方便以后分析各视图的单独性能
        return logits, la, lb


# ==============================================================================
# 评估函数：计算训练集/验证集/测试集的准确率
# ==============================================================================
@torch.no_grad()  # 这个装饰器告诉 PyTorch：下面的代码不需要计算梯度
                  # 评估时不需要更新参数，关掉梯度计算可以节省内存、加快速度
def eval_acc(model, data, edge_a, edge_b):
    """
    评估模型在三个数据集划分上的分类准确率。

    参数：
      model  : 训练好的（或训练中的）DualGCN 模型
      data   : 图数据对象，包含节点特征 x、标签 y、以及三个掩码 mask
      edge_a : 引文图的边索引
      edge_b : kNN图的边索引

    返回：
      [训练集准确率, 验证集准确率, 测试集准确率]，值域 [0, 1]
    """
    model.eval()  # 切换到"评估模式"：关闭 Dropout，BatchNorm 用统计值而非批次值

    # 前向传播，得到每个节点对 7 个类别的得分
    out, _, _ = model(data.x, edge_a, edge_b)

    # argmax(-1)：取每行最大值的索引，即预测的类别编号
    # 比如得分 [0.1, 0.3, 0.8, 0.2, 0.1, 0.05, 0.15] → 预测类别 2
    pred = out.argmax(dim=-1)

    accs = []
    # data.train_mask / val_mask / test_mask 是布尔张量（True/False）
    # True 表示该节点属于对应的分割，只在 True 的位置计算准确率
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        # pred[mask]：只取训练/验证/测试节点的预测结果
        # data.y[mask]：对应的真实标签
        # (pred[mask] == data.y[mask])：逐元素比较，正确则为1.0，错误则为0.0
        # .float().mean()：对布尔值取均值 = 准确率
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs  # 返回 [训练准确率, 验证准确率, 测试准确率]


# ==============================================================================
# 主函数：数据加载 → 构建图 → 初始化模型 → 训练循环
# ==============================================================================
def main():
    # ── 第一步：解析命令行参数 ────────────────────────────────────────────────
    # 通过 argparse，你可以在终端这样运行：
    #   python train_baseline.py --epochs 300 --lr 0.005 --k 15
    # 如果不传，就使用下面的 default 默认值
    p = argparse.ArgumentParser()
    p.add_argument("--k",      type=int,   default=10)    # kNN图中每个节点连接最近的 k 个邻居
    p.add_argument("--epochs", type=int,   default=200)   # 训练轮数（过一遍全部训练数据 = 1 epoch）
    p.add_argument("--hid",    type=int,   default=64)    # GCN 隐藏层维度
    p.add_argument("--lr",     type=float, default=0.01)  # 学习率：每次更新参数的步长大小
    p.add_argument("--wd",     type=float, default=5e-4)  # 权重衰减（L2正则化系数）：防止过拟合
    p.add_argument("--seed",   type=int,   default=7)     # 随机种子：固定后实验结果可复现
    args = p.parse_args()

    # ── 第二步：固定随机种子，保证每次运行结果一致 ────────────────────────────
    torch.manual_seed(args.seed)

    # ── 第三步：加载 Cora 数据集 ──────────────────────────────────────────────
    # Cora 是图神经网络领域最经典的 benchmark 数据集，包含：
    #   - 2708 个节点（论文）
    #   - 5429 条边（引用关系）
    #   - 每个节点有 1433 维特征（词袋向量，表示论文中出现了哪些词）
    #   - 7 个类别（机器学习的7个子领域）
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]  # 取出第一张（也是唯一一张）图

    # ── 第四步：构建两种不同的图（两个"视图"） ────────────────────────────────
    # 视图A：原始引文图，data.edge_index 是数据集自带的，形状 [2, 10858]（无向图，每条边存两次）
    edge_a = data.edge_index

    # 视图B：kNN特征相似度图
    # 对每个节点，找特征向量最相近的 k=10 个节点，连一条边
    # loop=False 表示不连自环（节点不连自己）
    # 结果形状：[2, 2708*10] = [2, 27080]
    edge_b = knn_graph(data.x, k=args.k, loop=False)

    # ── 第五步：初始化模型和优化器 ────────────────────────────────────────────
    # 创建 DualGCN 模型实例
    #   输入维度 = 每篇论文的特征数（1433）
    #   隐藏维度 = args.hid（64）
    #   输出维度 = 类别数（7）
    model = DualGCN(dataset.num_features, args.hid, dataset.num_classes)

    # Adam 优化器：比普通梯度下降更智能，会自适应调整每个参数的学习率
    # model.parameters()：把模型所有可学习的参数（权重矩阵等）交给优化器管理
    # weight_decay（权重衰减）：每次更新时把参数"往0拉"一点，起到正则化效果
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 用于记录验证集上的历史最佳准确率，以及对应时刻的测试集准确率
    best_val = 0.0
    best_test = 0.0

    # ── 第六步：训练循环（Training Loop）────────────────────────────────────
    # 每个 epoch 完整地过一遍训练数据，更新一次模型参数
    for epoch in range(1, args.epochs + 1):

        # 切换到训练模式（开启 Dropout）
        model.train()

        # 清空上一步积累的梯度（PyTorch 默认会累积梯度，每次都要手动清零）
        opt.zero_grad()

        # 前向传播：把数据喂给模型，得到预测结果
        out, _, _ = model(data.x, edge_a, edge_b)

        # 计算损失（Loss）：交叉熵损失，衡量预测结果与真实标签的差距
        # 只在训练集节点（data.train_mask 为 True 的节点）上计算损失
        # 损失越小说明预测越准确
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # 反向传播：自动计算每个参数对损失的梯度（偏导数）
        # 梯度告诉我们：调整哪个参数、往哪个方向调、调多少，才能让损失减小
        loss.backward()

        # 优化器更新参数：沿梯度下降方向走一步（步长 = 学习率）
        opt.step()

        # ── 评估当前模型性能 ────────────────────────────────────────────────
        train_acc, val_acc, test_acc = eval_acc(model, data, edge_a, edge_b)

        # 用验证集准确率来选模型：只保存验证集表现最好时对应的测试集准确率
        # 注意：不能直接用测试集选模型，那样会导致"测试集泄露"，结果不可信
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc  # 记录验证集最优时刻的测试集准确率

        # 每训练 20 个 epoch 打印一次进度
        # f-string 格式化：:03d 表示至少3位数字，:.4f 表示保留4位小数
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val {val_acc:.4f} | best_test {best_test:.4f}")

    # 训练结束，打印最终最佳测试集准确率
    print(f"Final best_test={best_test:.4f}")


# ==============================================================================
# Python 程序的标准入口
#
# __name__ == "__main__" 的意思是：
#   只有当这个文件被"直接运行"时（python train_baseline.py），才执行 main()
#   如果这个文件被其他文件"import"进去，就不会自动执行 main()
#   这是 Python 的惯用写法，几乎所有脚本都这样写
# ==============================================================================
if __name__ == "__main__":
    main()