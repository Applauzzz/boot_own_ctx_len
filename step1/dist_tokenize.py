import os, typer, torch, json
import torch.distributed as dist
from streaming import StreamingDataset, MDSWriter
from transformers import AutoTokenizer
from tqdm import tqdm
import array

app = typer.Typer()

def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def encode_u32(tokens):
    arr = array.array('I', tokens)
    return arr.tobytes()

@app.command()
def main(
    text_mds: str = typer.Option(...),
    out_mds: str  = typer.Option(...),
    model: str    = typer.Option("intfloat/e5-mistral-7b-instruct"),
    size_per_writer: int = typer.Option(512),
):
    rank, world, local_rank = setup_dist()
    tok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    my_out = os.path.join(out_mds, f"part_rank{rank:05d}")
    os.makedirs(my_out, exist_ok=True)

    ds = StreamingDataset(local=text_mds, shuffle=False, split=None, batch_size=1)
    columns = {'idx': 'int', 'tokens': 'bytes', 'length': 'int'}
    SIZE_LIMIT = size_per_writer * 1024 * 1024
    with MDSWriter(
        out=my_out,
        columns={'idx': 'int', 'tokens': 'bytes', 'length': 'int'},
        compression='zstd',
        hashes=['sha1'],
        size_limit=SIZE_LIMIT, 
    ) as w:
        for i, rec in enumerate(tqdm(ds, desc=f"rank{rank} tokenizing", disable=rank!=0)):
            # mds应该隐式读取了节点的rank
            # if i % world != rank:
            #     continue
            idx = int(rec.get('idx', -1))
            text = rec.get('text', '')
            if idx < 0 or not text:
                continue
            enc = tok(text, add_special_tokens=False, return_attention_mask=False, return_tensors=None)
            ids = enc['input_ids']
            # print(enc['input_ids'])
            w.write({'idx': idx, 'tokens': encode_u32(ids), 'length': len(ids)})

    dist.barrier()
    if rank == 0:
        print("[tokenize] All ranks done. Now run merge_index(out_mds) on the directory root to unify shards.")

if __name__ == "__main__":
    app()
