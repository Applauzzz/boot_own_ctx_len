# -*- coding: utf-8 -*-
"""
B2-B5: 检索 + QFS递归摘要 + 最终回答 + 训练样本构造（零通信分布式）
- 读取 B1 产出的 instruction-MDS
- 用 E5 向量化 search_queries 并在 FAISS 上检索 Top-K 文档
- 对检索文档做 4k 分块 + QFS（递归压缩到目标预算）
- 用（指令 + 全部摘要）生成最终回答
- 写训练样本：输入=指令+原始文档拼接（不含摘要），输出=最终回答；另存 metadata

  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
  torchrun --nproc_per_node=7 \
  python step_b2_b5_retrieve_qfs_answer_make_trainset.py \
    --inst-mds /data/mds_instructions \
    --text-mds /data/mds_text \
    --faiss-index /data/faiss/faiss.index \
    --faiss-ids /data/faiss/ids.txt \
    --out-mds /data/mds_train_samples \
    --retriever-model intfloat/e5-mistral-7b-instruct \
    --summarizer-model Qwen/Qwen2.5-7B-Instruct \
    --answer-model Qwen/Qwen2.5-7B-Instruct \
    --topk 12 \
    --per-query-topk 8 \
    --chunk-tokens 4096 \
    --summary-budget 4096 \
    --answer-max-new 1536
"""
import os, re, io, json, time, sqlite3, typing as T
import orjson
import numpy as np
import typer
from tqdm import tqdm

import torch
from streaming import StreamingDataset, MDSWriter
from streaming.base.util import merge_index
import faiss

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM
)


def setup_rank_env():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def kv_path(root: str) -> str:
    return os.path.join(root, "idx2text.sqlite")

def ensure_kv_from_text_mds(db_path: str, text_mds: str):
    if os.path.exists(db_path):
        return
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS kv (idx INTEGER PRIMARY KEY, text TEXT)")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = WAL;")

    ds = StreamingDataset(local=text_mds, batch_size=1, shuffle=False, split=None)
    buf = []
    BATCH = 5000
    for rec in tqdm(ds, desc="[KV] build idx->text"):
        idx = int(rec.get("idx", -1))
        text = rec.get("text", "")
        if idx >= 0 and text:
            buf.append((idx, text))
            if len(buf) >= BATCH:
                cur.executemany("INSERT OR REPLACE INTO kv (idx, text) VALUES (?,?)", buf)
                conn.commit()
                buf.clear()
    if buf:
        cur.executemany("INSERT OR REPLACE INTO kv (idx, text) VALUES (?,?)", buf)
        conn.commit()
    cur.close(); conn.close()

def kv_fetch_texts(db_path: str, ids: T.List[int]) -> T.List[str]:
    if not ids:
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    qmarks = ",".join("?" for _ in ids)
    cur.execute(f"SELECT idx, text FROM kv WHERE idx IN ({qmarks})", ids)
    got = dict(cur.fetchall())
    cur.close(); conn.close()
    return [got.get(i, "") for i in ids]

import torch.nn.functional as F

@torch.no_grad()
def _last_token_pool(last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    # 与建库一致的实现（left/right padding 兼容）
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        reps = last_hidden_states[:, -1]
    else:
        seq_len = attention_mask.sum(dim=1) - 1
        reps = last_hidden_states[
            torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), seq_len
        ]
    return F.normalize(reps, p=2, dim=-1)

def e5_format_query(q: str, task: str = "Given a web search query, retrieve relevant passages that answer the query") -> str:
    return f"Instruct: {task}\nQuery: {q}"

@torch.no_grad()
def e5_embed_queries(model, tokenizer, queries: T.List[str], device: torch.device, dtype=None) -> np.ndarray:
    texts = [e5_format_query(q) for q in queries]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=4096, return_tensors="pt").to(device)
    out = model(**enc)
    vec = _last_token_pool(out.last_hidden_state, enc["attention_mask"])  # 与库一致
    return vec.float().cpu().numpy()


class FaissRetriever:
    def __init__(self, index_path: str, metric: str = "ip", nprobe: int = 32):
        self.index = faiss.read_index(index_path)
        self.index.nprobe = nprobe
        self.metric = metric.lower()

    def search(self, Q: np.ndarray, topk: int):
        D, I = self.index.search(Q, topk)
        # I 直接是 doc idx，无需映射
        return D, I

class HFChat:
    def __init__(self, model_name: str, dtype="auto"):
        torch_dtype = {
            "auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32
        }.get(dtype, None)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def chat(self, system: str | None, user: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tok([text], return_tensors="pt").to(self.mdl.device)
        out = self.mdl.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        new_tokens = [o[len(i):] for i, o in zip(model_inputs.input_ids, out)]
        return self.tok.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def token_len(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])

    def cut_to(self, text: str, max_tokens: int) -> str:
        ids = self.tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        return self.tok.decode(ids[:max_tokens], skip_special_tokens=True)

# ---------------------------
# QFS Prompt & 递归压缩
# ---------------------------
QFS_SYSTEM = (
    "You are a helpful query-focused summarization agent. "
    "Given a user instruction and a document chunk, produce a concise summary "
    "that ONLY contains information relevant to answering the instruction. "
    "Avoid unrelated details."
)

QFS_TMPL = """# Instruction
{inst}

# Document Chunk
{doc}

# Task
Write a concise query-focused summary of the chunk that only keeps information directly relevant to the Instruction.
Do NOT include meta text. Keep it faithful and specific.
"""

FINAL_SYSTEM = (
    "You are a helpful long-context assistant. "
    "Given an instruction and a set of summaries distilled from multiple documents, "
    "synthesize a comprehensive, well-structured answer. "
    "Cite facts across summaries coherently; avoid hallucinations."
)

FINAL_TMPL = """# Instruction
{inst}

# Summaries
{summaries}

# Task
Write the final answer. Be concise but complete. Use bullet points or sections if helpful.
"""

def split_to_chunks(text: str, tok: AutoTokenizer, max_tokens: int) -> T.List[str]:
    ids = tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    chunks = []
    for i in range(0, len(ids), max_tokens):
        seg = tok.decode(ids[i:i+max_tokens], skip_special_tokens=True)
        if seg.strip():
            chunks.append(seg)
    return chunks

def qfs_once(chat: HFChat, inst: str, doc_chunk: str, max_new_tokens=384, temperature=0.3) -> str:
    prompt = QFS_TMPL.format(inst=inst, doc=doc_chunk)
    return chat.chat(QFS_SYSTEM, prompt, max_new_tokens=max_new_tokens, temperature=temperature).strip()

def recursive_qfs(chat: HFChat, inst: str, docs: T.List[str], chunk_tokens: int, summary_budget: int) -> str:
    """
    递归压缩：对每个doc分块→QFS；把所有chunk-summary拼接；若总token超预算→再作为“文档集合”继续QFS。
    """
    tok = chat.tok
    # 一级：对原始 doc 分块并摘要
    mini_summaries = []
    for d in docs:
        for ch in split_to_chunks(d, tok, chunk_tokens):
            s = qfs_once(chat, inst, ch)
            if s:
                mini_summaries.append(s)

    joined = "\n\n".join(f"- {s}" for s in mini_summaries)
    if chat.token_len(joined) <= summary_budget:
        return joined

    while chat.token_len(joined) > summary_budget:
        # 再次切块
        chunks = split_to_chunks(joined, tok, max_tokens=min(1024, chunk_tokens))
        next_summaries = [qfs_once(chat, inst, c, max_new_tokens=256, temperature=0.3) for c in chunks]
        joined = "\n\n".join(f"- {s}" for s in next_summaries if s.strip())

    return joined


def json_dumps(obj) -> str:
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")

# ---------------------------
# 主流程
# ---------------------------
app = typer.Typer(help="B2-B5 检索+QFS+最终回答+训练样本构造")

@app.command()
def main(
    inst_mds: str = typer.Option(..., help="B1 产出的 instruction-MDS"),
    text_mds: str = typer.Option(..., help="原始语料 text-MDS（含 idx,text）"),
    faiss_index: str = typer.Option(..., help="FAISS 索引路径"),

    out_mds: str = typer.Option(..., help="输出训练样本 MDS 根目录"),
    database_path: str = typer.Option("", help="SQLite KV 数据库路径"),
    retriever_model: str = typer.Option("intfloat/e5-mistral-7b-instruct"),
    summarizer_model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct"),
    answer_model: str     = typer.Option("Qwen/Qwen2.5-7B-Instruct"),

    topk: int = typer.Option(12, help="合并后的最终Top-K文档"),
    per_query_topk: int = typer.Option(8, help="每条搜索query返回多少，再合并去重"),
    n_queries_limit: int = typer.Option(6, help="每条指令最多用多少 query"),
    chunk_tokens: int = typer.Option(4096, help="文档分块大小（tokens）"),
    summary_budget: int = typer.Option(4096, help="合并摘要总预算（tokens）"),

    retriever_dtype: str = typer.Option("fp16"),      # E5 dtype
    summarizer_dtype: str = typer.Option("fp16"),
    answer_dtype: str     = typer.Option("fp16"),

    answer_temperature: float = typer.Option(0.7),
    answer_max_new: int = typer.Option(1536),

    per_rank_buffer: int = typer.Option(16, help="写盘缓冲"),
    compression: str = typer.Option("zstd"),
):
    rank, world, local_rank = setup_rank_env()
    os.makedirs(out_mds, exist_ok=True)


    dbp = database_path if database_path else kv_path(out_mds)
    if rank == 0:
        ensure_kv_from_text_mds(dbp, text_mds)
    time.sleep(2)

    ds_inst = StreamingDataset(local=inst_mds, batch_size=1, shuffle=False, split=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu",
                          local_rank if torch.cuda.is_available() else 0)
    e5_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "auto": None}.get(retriever_dtype, None)

    retriever = FaissRetriever(faiss_index, metric="ip", nprobe=64)

    # === Summarizer / Answer LLM ===
    summarizer = HFChat(summarizer_model, dtype=summarizer_dtype)
    answer_llm = HFChat(answer_model, dtype=answer_dtype)

    # print("start sanity check")
    # Q1 = e5_embed_queries(retriever_model, ["what is quantum annealing?"], device, e5_dtype)
    # Q2 = e5_embed_queries(retriever_model, ["history of Tang dynasty"], device, e5_dtype)
    # D1, I1 = retriever.search(Q1, 10)
    # D2, I2 = retriever.search(Q2, 10)
    # print("I1:", I1[0].tolist())
    # print("I2:", I2[0].tolist())
    # print("overlap:", len(set(I1[0]) & set(I2[0])))
    # 期望 overlap 很小


    # === 输出 MDS ===
    my_out = os.path.join(out_mds, f"part_rank{rank:05d}")
    os.makedirs(my_out, exist_ok=True)
    done_dir = os.path.join(out_mds, "_done"); os.makedirs(done_dir, exist_ok=True)
    done_flag = os.path.join(done_dir, f"rank{rank:05d}.done")

    cols = {
        "inst_id": "int",
        "instruction": "str",
        "queries": "json",
        "retrieved_ids": "json",
        "train_input": "str",
        "train_output": "str",
    }

    def uniq_keep_best(pairs: T.List[T.Tuple[int, float]]) -> T.List[int]:
        # doc_id → max score
        best = {}
        for did, sc in pairs:
            if did < 0: continue
            if did not in best or sc > best[did]:
                best[did] = sc
        # 排序取前 topk
        return [k for k, _ in sorted(best.items(), key=lambda x: -x[1])[:topk]]

    def do_retrieve_for_instruction(ret_model, tok, inst: str, queries: T.List[str]) -> T.List[int]:
        queries = [q for q in queries[:n_queries_limit] if q.strip()]
        if not queries:
            queries = [inst]  # 兜底
        Q = e5_embed_queries(ret_model, tok, queries, device=device, dtype=e5_dtype)  # [nq, D]
        D, I = retriever.search(Q, per_query_topk)  # [nq, k]
        doc_ids = I       # [nq, k]
        pairs = []
        for qi in range(doc_ids.shape[0]):
            for ki in range(doc_ids.shape[1]):
                pairs.append((int(doc_ids[qi, ki]), float(D[qi, ki])))
        return uniq_keep_best(pairs)


    with MDSWriter(out=my_out, columns=cols, compression=compression, hashes=["sha1"]) as w:
        buf = []
        pbar = tqdm(desc=f"[rank{rank}] B2-B5", disable=(rank != 0))
        def setup_model_and_tokenizer(model_name: str, dtype="auto", device="cuda"):
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            dtype = dtype_map.get(dtype, torch.float16)
            mdl = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
            return mdl, tok

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank if torch.cuda.is_available() else 0)
        model, tokenizer = setup_model_and_tokenizer(retriever_model, dtype="fp16", device=device)

        for i, rec in enumerate(ds_inst):
            if i % world != rank:
                continue

            inst_id = int(rec.get("inst_id", i))
            instruction = str(rec.get("task_instruction", "") or "").strip()
            queries = rec.get("search_queries", []) or []
            if isinstance(queries, str):
                try:
                    queries = json.loads(queries)
                except Exception:
                    queries = [queries]

            doc_ids = do_retrieve_for_instruction(model, tokenizer, instruction, queries)

            raw_docs = kv_fetch_texts(dbp, doc_ids)
            paired = [(did, d) for did, d in zip(doc_ids, raw_docs) if d and d.strip()]
            if not paired:
                continue
            doc_ids = [x for x, _ in paired]
            raw_docs = [d for _, d in paired]

            summaries = recursive_qfs(
                summarizer, instruction, raw_docs, chunk_tokens=chunk_tokens, summary_budget=summary_budget
            )

            final_prompt = FINAL_TMPL.format(inst=instruction, summaries=summaries)
            answer = answer_llm.chat(
                FINAL_SYSTEM, final_prompt, max_new_tokens=answer_max_new, temperature=answer_temperature
            ).strip()

            train_input = (
                f"# Instruction\n{instruction}\n\n"
                f"# Retrieved Documents\n" +
                "\n\n".join([f"[DOC {j+1} | id={doc_ids[j]}]\n{raw_docs[j]}" for j in range(len(raw_docs))])
            )
            train_output = answer

            buf.append({
                "inst_id": inst_id,
                "instruction": instruction,
                "queries": queries,
                "retrieved_ids": doc_ids,
                "train_input": train_input,
                "train_output": train_output,
            })

            if len(buf) >= per_rank_buffer:
                for x in buf: w.write(x)
                buf.clear()

            pbar.update(1)

        if buf:
            for x in buf: w.write(x)
        pbar.close()

    # 标记完成并在 rank0 合并
    open(done_flag, "w").close()
    if rank == 0:
        import glob
        expect = {f"rank{i:05d}.done" for i in range(world)}
        while True:
            got = {os.path.basename(p) for p in glob.glob(os.path.join(done_dir, "rank*.done"))}
            if expect.issubset(got):
                break
            time.sleep(1.0)
        merge_index(out_mds)
        meta = {
            "retriever_model": retriever_model,
            "summarizer_model": summarizer_model,
            "answer_model": answer_model,
            "topk": topk,
            "per_query_topk": per_query_topk,
            "chunk_tokens": chunk_tokens,
            "summary_budget": summary_budget,
            "answer_max_new": answer_max_new
        }
        with open(os.path.join(out_mds, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ok] merged train MDS -> {out_mds}")


if __name__ == "__main__":
    app()
