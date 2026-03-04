import os
import torch
import torch.distributed as dist
from datetime import timedelta


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with NCCL requires CUDA, but CUDA is not available")
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid LOCAL_RANK={local_rank} for {torch.cuda.device_count()} visible CUDA devices"
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(hours=2),
        )
        dist.barrier()
        return rank

    # Single-process fallback (no DDP env).
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0
