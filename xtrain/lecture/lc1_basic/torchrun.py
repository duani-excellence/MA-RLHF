# torchrun --standalone --nnodes=1  --nproc-per-node="cpu" torchrun.py 

# torchrun --standalone --master-addr="127.0.0.1" --node-rank=0  --nnodes=2  --nproc-per-node="cpu" torchrun.py &
# torchrun --standalone --master-addr="127.0.0.1" --node-rank=1  --nnodes=2  --nproc-per-node="cpu" torchrun.py &

# import argparse
import os
# parser = argparse.ArgumentParser()
# parser.add_argument("--local-rank", "--local_rank", type=int)

local_rank = int(os.environ["LOCAL_RANK"])
print(local_rank)

if local_rank == 0:
    print('world_size:', os.environ["WORLD_SIZE"])