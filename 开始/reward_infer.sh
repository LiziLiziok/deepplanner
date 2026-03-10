CUDA_VISIBLE_DEVICES=0,1 python -m sglang_router.launch_server \
    --model-path /apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/Search-R1/model/Qwen2.5-3B-Instruct \
    --dp-size 2 \
    --host 0.0.0.0 \
    --port 30000
