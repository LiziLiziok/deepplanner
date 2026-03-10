save_path=/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/Search-R1/data
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/Search-R1/model/e5-base-v2

# 使用 python 直接启动（单 worker，适合 GPU 场景）
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 5 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
