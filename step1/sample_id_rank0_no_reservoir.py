import os, random, typer, array
from tqdm import tqdm
from streaming import StreamingDataset, MDSWriter

app = typer.Typer(help="从 tokens-MDS 直接筛选并写出子集 tokens-MDS（含 idx,tokens,length）")

def make_ds(local_dir: str, seed: int, batch_size: int = 1):
    return StreamingDataset(
        local=local_dir,
        batch_size=batch_size,
        shuffle=True,
        shuffle_seed=seed,
        split=None,
    )

@app.command()
def main(
    tokens_mds: str = typer.Option(..., help="全量 tokens-MDS 根目录（列：idx:int, tokens:bytes, length:int）"),
    out_mds:    str = typer.Option(..., help="输出 子集 tokens-MDS 根目录"),
    min_len:    int = typer.Option(2000, help="最小长度"),
    max_len:    int = typer.Option(32000, help="最大长度"),
    target:     int = typer.Option(10000, help="最多写出条数（写满即止）"),
    seed:       int = typer.Option(42),
    size_per_writer: int = typer.Option(512),
    compression: str   = typer.Option("zstd"),
):
    ds = make_ds(tokens_mds, seed=seed, batch_size=1)
    os.makedirs(out_mds, exist_ok=True)
    cols = {"idx": "int", "tokens": "bytes", "length": "int"}

    written = 0
    SIZE_LIMIT = size_per_writer * 1024 * 1024

    with MDSWriter(out=out_mds, columns=cols, compression=compression,
                   hashes=["sha1"], size_limit=SIZE_LIMIT) as w:
        for rec in tqdm(ds, desc="filter & write"):
            L = int(rec.get("length", 0))
            if not (min_len <= L <= max_len):
                continue
            idx = int(rec["idx"])
            w.write({"idx": idx, "tokens": rec["tokens"], "length": L})
            written += 1
            if written >= target:
                break

    print(f"[ok] 子集写出 {written} 条 → {out_mds}")

if __name__ == "__main__":
    app()
