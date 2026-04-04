import torch
import sys, os
from torch_geometric.data import Data, Batch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.edgemamba3 import EdgeMamba3

def test_vn():
    print("Testing VN initialization...")
    model = EdgeMamba3(
        domain="lrgb", node_in_dim=9, edge_in_dim=3,
        d_model=64, n_layers=2, d_state=16, mimo_rank=2,
        num_outputs=10, task_type="classification",
        use_virtual_node=True, use_mamba2=True # use mamba_ssm fallback safely if missing
    ).cuda()
    
    print("Creating dummy batches...")
    data1 = Data(x=torch.randn(10, 9), edge_index=torch.randint(0, 10, (2, 20)), edge_attr=torch.randn(20, 3)).cuda()
    data2 = Data(x=torch.randn(15, 9), edge_index=torch.randint(0, 15, (2, 30)), edge_attr=torch.randn(30, 3)).cuda()
    
    print("Testing single graph forward...")
    out_single = model._forward_lrgb_single(data1)
    print("Single graph out shape:", out_single.shape)
    
    # Batched
    batch = Batch.from_data_list([data1, data2]).cuda()
    print("Testing batched forward...")
    out_batched = model._forward_lrgb_batched(batch)
    print("Batched out shape:", out_batched.shape)
    
    print("SUCCESS")

if __name__ == "__main__":
    print("CUDA is available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        test_vn()
