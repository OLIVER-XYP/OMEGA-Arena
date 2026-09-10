import os
import torch
import torch.distributed as dist

def run_worker(rank, world_size):
    # 注意：这里使用 NCCL 后端，必须指定 init_method
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",  # 主节点地址
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    
    # 简单 All-Reduce 通信测试
    tensor = torch.ones(10).cuda() * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} 通信结果: {tensor[0].item()}")  # 期望结果为 (0+1+...+7)=28
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 8
    # 使用 multiprocessing 启动多进程
    torch.multiprocessing.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)