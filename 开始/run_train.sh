# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs

# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash /apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/examples/sglang_multiturn/search_r1_like/search_r1_run.sh trainer.experiment_name=search_r1_classic-$(now) > logs/searchR1-classic$(now).log 2>&1 &
