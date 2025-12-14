# run on 8xH20
# make sure your current working directory is the root of the project
# 记得先配置 wandb login
# 本脚本中，设置    trainer.val_before_train=True 以及 trainer.val_only=True ，只用于评测；如果训练，则删除这两个参数。
set -x

ulimit -n 65535
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_MODEL='/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/models/Qwen2.5-7B-Instruct'
export DATA_DIR='/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/data/EviNoteRAG_nq_hotpotqa_train_and_test_data'

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"


echo "开始--"
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.agent.default_agent_loop=deep_planner_agent \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl_deepplanner' \
    trainer.experiment_name='qwen2.5-3b-instruct_function_rm-search-async-sgl-multi-w-searchtool-verify-n16_deepplanner' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    data.train_files=$DATA_DIR/m_test_prepared_total.parquet \
    data.val_files=$DATA_DIR/m_test_prepared_total.parquet  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1 $@

