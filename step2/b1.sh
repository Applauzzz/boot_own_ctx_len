rm -r /home/zhliu/database/mds_instructions
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 torchrun --nproc_per_node=1 \
python step_b1_make_instructions.py \
--text-mds /home/zhliu/database/mds_wiki \
--out-mds  /home/zhliu/database/mds_instructions \
--n-instructions 20 \
--llm-backend hf \
--llm-model /home/zhliu/model_base/Qwen2.5-7B-Instruct \
--chunk-tokenizer /home/zhliu/model_base/Qwen2.5-7B-Instruct \
--rnd-chunk-tokens 1024 \
--per-rank-buffer 64