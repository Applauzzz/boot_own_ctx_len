# -*- coding: utf-8 -*-
"""
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  torchrun --nproc_per_node=7 \
  python step_b1_make_instructions.py \
    --text-mds /data/mds_text \
    --out-mds  /data/mds_instructions \
    --n-instructions 200000 \
    --llm-backend hf \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --chunk-tokenizer intfloat/e5-mistral-7b-instruct \
    --rnd-chunk-tokens 256 \
    --per-rank-buffer 64
"""
import os, re, json, random, time, typing as T
import orjson
import typer
from tqdm import tqdm

import torch
from streaming import StreamingDataset, MDSWriter
from streaming.base.util import merge_index
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_rank_env():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


class TokLimiter:
    def __init__(self, name: str | None):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True) if name else None

    def trim(self, text: str, max_tokens: int) -> str:
        if not self.tok or max_tokens <= 0:
            return text
        ids = self.tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]

        return self.tok.decode(ids, skip_special_tokens=True)


SYSTEM_PROMPT = (
    "You are a helpful data synthesis assistant. "
    "Given a random text chunk, you will brainstorm a task/question that requires integrating multiple pieces "
    "of information and produce a JSON spec with a task instruction and diverse search queries. "
    "Return ONLY a valid JSON object."
)

INSTR_PROMPT_TMPL = """{rnd_chunk}

# Brainstorm a potentially useful task or question that requires integrating multiple pieces of information.

Return a JSON with keys:
- "task_instruction": string
- "search_queries": [string, ...]

Guidelines:
- Diverse domains and difficulty.
- Requires logical/common-sense/mathematical reasoning.
- Feasible for a text-only AI model (no visuals).
- Queries cover distinct aspects.

Only return the JSON object, nothing else.
"""

def build_instruction_prompt(rnd_chunk: str) -> str:
    return INSTR_PROMPT_TMPL.format(rnd_chunk=rnd_chunk)


class LLMBackend:
    def generate_json(self, system_prompt: str | None, user_prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        raise NotImplementedError

# ---------- HF (Transformers) ----------
class HFBackend(LLMBackend):
    def __init__(self, model_name: str, dtype: str = "auto"):
        torch_dtype = {
            "auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32
        }.get(dtype, None)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    def _format_chat(self, system: str | None, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate_json(self, system_prompt: str | None, user_prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        text = self._format_chat(system_prompt, user_prompt)
        model_inputs = self.tok([text], return_tensors="pt").to(self.mdl.device)
        gen_out = self.mdl.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        new_tokens = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, gen_out)
        ]
        text_out = self.tok.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return robust_json_extract(text_out)

class OpenAIBackend(LLMBackend):
    def __init__(self, model_name: str):
        import openai  # type: ignore
        self.client = openai.OpenAI()
        self.model = model_name

    def generate_json(self, system_prompt: str | None, user_prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt + "\n\nReturn ONLY a valid JSON object."})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        return robust_json_extract(text)

class VLLMBackend(LLMBackend):
    def __init__(self, url: str, model_name: str):
        import requests  # type: ignore
        self.requests = requests
        self.url = url
        self.model = model_name

    def generate_json(self, system_prompt: str | None, user_prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        payload = {
            "model": self.model,
            "prompt": (system_prompt + "\n\n" if system_prompt else "") +
                      user_prompt + "\n\nReturn ONLY a valid JSON object.",
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = self.requests.post(self.url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        text = data.get("text") or data.get("choices", [{}])[0].get("text", "")
        return robust_json_extract(text)

JSON_BLOCK = re.compile(r"\{.*\}", re.S)

def robust_json_extract(text: str) -> dict:

    m = JSON_BLOCK.search(text)
    snippet = (m.group(0) if m else text).strip()
    print(text)
    for loader in (json.loads, orjson.loads):
        try:
            obj = loader(snippet)
            break
        except Exception:
            obj = None
    if obj is None:
        # 兜底
        obj = {
            "task_instruction": "Research and compare two related topics, then provide a structured conclusion.",
            "search_queries": [
                "topic A overview", "topic B overview", "comparison A vs B", "recent studies about A and B"
            ]
        }

    task = obj.get("task_instruction") or obj.get("instruction") or ""
    queries = obj.get("search_queries") or obj.get("queries") or []
    if not isinstance(queries, list):
        queries = [str(queries)]
    queries = [q for q in (str(x).strip() for x in queries) if q]
    while len(queries) < 4:
        queries.append("additional related query")
    return {"task_instruction": str(task).strip(), "search_queries": queries[:12]}


def iter_random_chunks(ds: StreamingDataset, tok_limiter: TokLimiter, rnd_chunk_tokens: int, seed: int):
    rnd = random.Random(seed)
    for rec in ds:
        text = rec.get("text", "")
        if not text:
            continue
        if len(text) > 2000:
            start = rnd.randrange(0, max(1, len(text) - 1000))
            text = text[start:start + 1000]
        yield tok_limiter.trim(text, rnd_chunk_tokens)


app = typer.Typer(help="B1: 生成 Instruction（带随机语料前缀）")

@app.command()
def main(
    text_mds: str = typer.Option(..., help="text-MDS 根目录（列：idx:int, text:str）"),
    out_mds:  str = typer.Option(..., help="输出 instructions-MDS"),
    n_instructions: int = typer.Option(200000, help="全局生成条数"),
    llm_backend: str = typer.Option("hf", help="hf | openai | vllm"),
    llm_model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct"),
    vllm_url: str = typer.Option("http://127.0.0.1:8000/generate"),
    chunk_tokenizer: str = typer.Option("intfloat/e5-mistral-7b-instruct", help="用于前缀截断的分词器"),
    rnd_chunk_tokens: int = typer.Option(256, help="随机前缀 token 上限"),
    temperature: float = typer.Option(0.7),
    max_new_tokens: int = typer.Option(512),
    seed: int = typer.Option(2025),
    per_rank_buffer: int = typer.Option(64, help="写盘缓冲条数"),
    compression: str = typer.Option("zstd"),
):
    rank, world, local_rank = setup_rank_env()
    os.makedirs(out_mds, exist_ok=True)

    per_rank = (n_instructions + world - 1) // world
    start_id = rank * per_rank
    end_id   = min(n_instructions, (rank + 1) * per_rank)

    ds = StreamingDataset(local=text_mds, batch_size=1, shuffle=True, split=None)
    tok_limiter = TokLimiter(chunk_tokenizer)

    if llm_backend == "hf":
        llm = HFBackend(llm_model, dtype="fp16" if torch.cuda.is_available() else "fp32")
    elif llm_backend == "openai":
        llm = OpenAIBackend(llm_model)
    elif llm_backend == "vllm":
        llm = VLLMBackend(vllm_url, llm_model)
    else:
        raise ValueError("llm-backend must be one of {hf, openai, vllm}")

    my_out = os.path.join(out_mds, f"part_rank{rank:05d}")
    os.makedirs(my_out, exist_ok=True)
    done_dir = os.path.join(out_mds, "_done"); os.makedirs(done_dir, exist_ok=True)
    done_flag = os.path.join(done_dir, f"rank{rank:05d}.done")

    cols = {"inst_id":"int", "rand_chunk":"str", "task_instruction":"str", "search_queries":"json"}

    chunk_iter = iter_random_chunks(ds, tok_limiter, rnd_chunk_tokens, seed + rank)

    def gen_one_payload() -> dict:
        rnd_chunk = next(chunk_iter)
        prompt = build_instruction_prompt(rnd_chunk)
        obj = llm.generate_json(SYSTEM_PROMPT, prompt, max_tokens=max_new_tokens, temperature=temperature)
        obj["rand_chunk"] = rnd_chunk
        return obj

    # 写盘
    with MDSWriter(out=my_out, columns=cols, compression=compression, hashes=["sha1"]) as w:
        buf: list[dict] = []
        pbar = tqdm(total=(end_id - start_id), desc=f"[rank{rank}] make inst", disable=(rank!=0))
        inst_id = start_id
        while inst_id < end_id:
            try:
                payload = gen_one_payload()
                payload["inst_id"] = inst_id

                task = (payload.get("task_instruction") or "").strip()
                queries = payload.get("search_queries", [])
                queries = [q for q in (str(x).strip() for x in queries) if q][:12]
                if not task or len(queries) < 4:
                    task = task or "Analyze and synthesize information from multiple sources."
                    while len(queries) < 4:
                        queries.append("related query")
                buf.append({
                    "inst_id": inst_id,
                    "rand_chunk": payload["rand_chunk"],
                    "task_instruction": task,
                    "search_queries": queries,
                })

                inst_id += 1
                pbar.update(1)

                if len(buf) >= per_rank_buffer:
                    for x in buf: w.write(x)
                    buf.clear()

            except StopIteration:

                ds = StreamingDataset(local=text_mds, batch_size=1, shuffle=True, split=None)
                chunk_iter = iter_random_chunks(ds, tok_limiter, rnd_chunk_tokens, seed + rank + int(time.time()))
            except Exception as e:
                print(f"[rank{rank}] warn: generation failed: {e}")
                continue

        if buf:
            for x in buf: w.write(x)
        pbar.close()

    open(done_flag, "w").close()

    # rank0 合并
    if rank == 0:
        import glob
        expected = {f"rank{i:05d}.done" for i in range(world)}
        while True:
            got = {os.path.basename(p) for p in glob.glob(os.path.join(done_dir, "rank*.done"))}
            if expected.issubset(got):
                break
            time.sleep(1.5)

        merge_index(out_mds)
        meta = {
            "backend": llm_backend,
            "model": llm_model,
            "rnd_chunk_tokens": rnd_chunk_tokens,
            "chunk_tokenizer": chunk_tokenizer,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "n_instructions": n_instructions,
        }
        with open(os.path.join(out_mds, "instructions_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ok] instructions merged -> {out_mds}, total target={n_instructions}, per_rank~{per_rank}")

if __name__ == "__main__":
    app()
