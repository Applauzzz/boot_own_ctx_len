import os, json, typer, array
import numpy as np
import faiss
from streaming import StreamingDataset
from tqdm import tqdm

app = typer.Typer(help="从 embeddings-MDS（列: idx:int, embedding:bytes, dim:int）训练并构建 FAISS")

def bytes_to_float32_vec(b: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(b, dtype=np.float32, count=dim)
    return v.copy()

def stream_embeddings_from_mds(mds_dir, limit=None):
    ds = StreamingDataset(local=mds_dir, batch_size=1, shuffle=False, split=None)
    cnt = 0
    D = None
    for rec in ds:
        idx = int(rec.get("idx", -1))
        dim = int(rec.get("dim", -1))
        emb_b = rec.get("embedding", None)
        if idx < 0 or dim <= 0 or not emb_b:
            continue
        if D is None:
            D = dim
        elif D != dim:
            raise RuntimeError(f"dim mismatch: seen {D} vs {dim}")
        emb = bytes_to_float32_vec(emb_b, dim)
        yield idx, emb
        cnt += 1
        if limit is not None and cnt >= limit:
            break

def collect_train_matrix(train_mds, max_train=None):
    ids = []
    embs = []
    D = None
    for idx, v in tqdm(stream_embeddings_from_mds(train_mds, limit=max_train), desc="[train] load"):
        if D is None:
            D = v.shape[0]
        embs.append(v[None, :])
        ids.append(idx)
    if not embs:
        raise RuntimeError("no training vectors found")
    X = np.concatenate(embs, axis=0).astype("float32")
    return X, ids

@app.command()
def main(
    train_mds: str = typer.Option(..., help="用于训练的采样向量 MDS 目录（列: idx, embedding, dim）"),
    full_mds:  str = typer.Option(..., help="全量向量 MDS 目录（列: idx, embedding, dim）"),
    out_dir:   str = typer.Option(..., help="输出目录"),
    nlist:     int = typer.Option(4096),
    nprobe:    int = typer.Option(32),
    use_pq:    bool= typer.Option(False),
    M:         int = typer.Option(64),
    nbits:     int = typer.Option(8),
    metric:    str = typer.Option("ip", help="ip 或 l2"),
    max_train: int = typer.Option(1_000_000, help="训练向量上限"),
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 取训练集矩阵
    Xtr, tr_ids = collect_train_matrix(train_mds, max_train=max_train)
    dim = Xtr.shape[1]

    # 2) 归一化 + 选择度量
    if metric.lower() == "ip":
        faiss.normalize_L2(Xtr)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlatIP(dim)
    elif metric.lower() == "l2":
        faiss_metric = faiss.METRIC_L2
        quantizer = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    # 3) 建索引 & 训练
    if use_pq:
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, nbits, faiss_metric)
    else:
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
    print(f"[train] nlist={nlist}, use_pq={use_pq}, dim={dim}, train_size={len(Xtr):,}")
    index.train(Xtr)
    del Xtr

    # 4) 流式把全量向量 add_with_ids 进来
    added = 0
    buf_vecs = []
    buf_ids  = []
    BATCH = 100_000  # 累积到一定规模再一次性 add，减少调用开销
    for idx, v in tqdm(stream_embeddings_from_mds(full_mds), desc="[add] full"):
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            # 单向量归一化
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = (v / nrm).astype("float32", copy=False)
        buf_vecs.append(v[None, :])
        buf_ids.append(idx)
        if len(buf_ids) >= BATCH:
            V = np.concatenate(buf_vecs, axis=0)
            I = np.asarray(buf_ids, dtype="int64")
            index.add_with_ids(V, I)
            added += len(I)
            buf_vecs.clear(); buf_ids.clear()
            if added % 1_000_000 == 0:
                print(f"[add] added {added:,}")

    if buf_ids:
        V = np.concatenate(buf_vecs, axis=0)
        I = np.asarray(buf_ids, dtype="int64")
        index.add_with_ids(V, I)
        added += len(I)

    index.nprobe = nprobe
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "dim": dim, "metric": metric.lower(), "nlist": nlist, "nprobe": nprobe,
            "use_pq": use_pq, "M": M, "nbits": nbits
        }, f, indent=2)
    print(f"[ok] index @ {out_dir}, ntotal={index.ntotal:,}, added={added:,}")

if __name__ == "__main__":
    app()
