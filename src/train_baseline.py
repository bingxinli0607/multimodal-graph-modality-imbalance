import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import knn_graph

class DualGCN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        super().__init__()
        self.gcn1_a = GCNConv(in_dim, hid)
        self.gcn2_a = GCNConv(hid, out_dim)

        self.gcn1_b = GCNConv(in_dim, hid)
        self.gcn2_b = GCNConv(hid, out_dim)

        self.dropout = dropout

    def forward(self, x, edge_a, edge_b):
        # View A: citation graph
        xa = F.relu(self.gcn1_a(x, edge_a))
        xa = F.dropout(xa, p=self.dropout, training=self.training)
        la = self.gcn2_a(xa, edge_a)

        # View B: kNN graph from features
        xb = F.relu(self.gcn1_b(x, edge_b))
        xb = F.dropout(xb, p=self.dropout, training=self.training)
        lb = self.gcn2_b(xb, edge_b)

        # naive fusion
        logits = 0.5 * la + 0.5 * lb
        return logits, la, lb

@torch.no_grad()
def eval_acc(model, data, edge_a, edge_b):
    model.eval()
    out, _, _ = model(data.x, edge_a, edge_b)
    pred = out.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs  # train/val/test

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hid", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]

    edge_a = data.edge_index
    edge_b = knn_graph(data.x, k=args.k, loop=False)

    model = DualGCN(dataset.num_features, args.hid, dataset.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        out, _, _ = model(data.x, edge_a, edge_b)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        train_acc, val_acc, test_acc = eval_acc(model, data, edge_a, edge_b)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val {val_acc:.4f} | best_test {best_test:.4f}")

    print(f"Final best_test={best_test:.4f}")

if __name__ == "__main__":
    main()