# embed_from_tokens_mds.py
import os, json, array, typer, numpy as np, torch
from tqdm import tqdm
from streaming import StreamingDataset, MDSWriter
from streaming.base.util import merge_index
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

app = typer.Typer(help="从 tokens-MDS 生成 embeddings-MDS（零通信，每rank独立落盘）")

def setup_rank_env():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank

@torch.no_grad()
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        reps = last_hidden_states[:, -1]
    else:
        seq_len = attention_mask.sum(dim=1) - 1
        reps = last_hidden_states[
            torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), seq_len
        ]
    return F.normalize(reps, p=2, dim=-1)

def bytes_to_u32_list(b: bytes):
    arr = array.array("I"); arr.frombytes(b); return list(arr)

def pad_trunc_batch(list_of_token_ids, pad_id, max_len):
    clipped = [ids[:max_len] for ids in list_of_token_ids]
    L = max((len(x) for x in clipped), default=0)
    L = min(L, max_len)
    # input_ids
    padded = [ids + [pad_id]*(L - len(ids)) for ids in clipped]
    input_ids = torch.tensor(padded, dtype=torch.long)
    # attention_mask
    attn = [[1]*len(ids) + [0]*(L-len(ids)) for ids in clipped]
    attention_mask = torch.tensor(attn, dtype=torch.long)
    return input_ids, attention_mask

@app.command()
def main(
    tokens_mds: str  = typer.Option(..., help="输入 tokens-MDS 根目录（列：idx,tokens,length）"),
    out_mds:    str  = typer.Option(..., help="输出 embeddings-MDS 根目录"),
    model:      str  = typer.Option("intfloat/e5-mistral-7b-instruct"),
    batch_size: int  = typer.Option(8),
    max_length: int  = typer.Option(4096),
    torch_dtype: str = typer.Option("fp16", help="auto|bf16|fp16|fp32"),
    doc_prefix: str  = typer.Option("", help='可选文本前缀（一般doc不加；若需要试验，可用 "passage: "）'),
    size_limit_mb: int = typer.Option(512, help="单writer分片上限(MiB)"),
    compression:   str = typer.Option("zstd"),
):
    rank, world, local_rank = setup_rank_env()
    os.makedirs(out_mds, exist_ok=True)

    # 模型
    tok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.float16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank if torch.cuda.is_available() else 0)
    mdl = AutoModel.from_pretrained(model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    prefix_ids = tok(doc_prefix, add_special_tokens=False, return_attention_mask=False)["input_ids"] if doc_prefix else []

    ds = StreamingDataset(local=tokens_mds, batch_size=1, shuffle=False, split=None)
    import random
    rnd = random.Random(rank + 2025)
    sample_ratio = 1 
    my_out = os.path.join(out_mds, f"part_rank{rank:05d}")
    os.makedirs(my_out, exist_ok=True)
    cols = {"idx": "int", "embedding": "bytes", "dim": "int"}
    size_limit = size_limit_mb * 1024 * 1024
    done_dir = os.path.join(out_mds, "_done")
    os.makedirs(done_dir, exist_ok=True)
    done_flag = os.path.join(done_dir, f"rank{rank:05d}.done")

    def embed_and_pack(batch_indices, batch_token_lists):
        if not batch_indices:
            return []
        input_ids, attn = pad_trunc_batch(batch_token_lists, pad_id=pad_id, max_len=max_length)
        input_ids = input_ids.to(device); attn = attn.to(device)
        with torch.no_grad():
            out = mdl(input_ids=input_ids, attention_mask=attn)
            emb = last_token_pool(out.last_hidden_state, attn).float().cpu().numpy()  # [B, D]
        D = int(emb.shape[1])
        return [{"idx": int(idx), "embedding": emb[i].tobytes(), "dim": D}
                for i, idx in enumerate(batch_indices)]

    with MDSWriter(out=my_out, columns=cols, compression=compression,
                   hashes=["sha1"], size_limit=size_limit) as writer:
        buf_idx, buf_tok = [], []
        for i, rec in enumerate(tqdm(ds, desc=f"[rank{rank}] embed", disable=(rank != 0))):
            # if i % world != rank:
            #     continue
            if rnd.random() > sample_ratio:
                continue
            idx = int(rec.get("idx", -1))
            if idx < 0:
                continue
            ids = bytes_to_u32_list(rec["tokens"])
            if prefix_ids:
                ids = prefix_ids + ids
            buf_idx.append(idx)
            buf_tok.append(ids)
            if len(buf_idx) >= batch_size:
                for s in embed_and_pack(buf_idx, buf_tok):
                    writer.write(s)
                buf_idx, buf_tok = [], []

        if buf_idx:
            for s in embed_and_pack(buf_idx, buf_tok):
                writer.write(s)


    open(done_flag, "w").close()

    if rank == 0:
        import time, glob
        expected = {f"rank{i:05d}.done" for i in range(world)}
        while True:
            got = {os.path.basename(p) for p in glob.glob(os.path.join(done_dir, "rank*.done"))}
            if expected.issubset(got):
                break
            time.sleep(2)
        merge_index(out_mds)
        meta = {
            "model": model,
            "pooling": "last_token",
            "doc_prefix": doc_prefix,
            "max_length": max_length,
            "dtype": str(dtype),
        }
        with open(os.path.join(out_mds, "embed_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ok] embeddings-MDS 合并完成：{out_mds}")

if __name__ == "__main__":
    app()
