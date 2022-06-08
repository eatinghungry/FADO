CUDA_VISIBLE_DEVICES=0 python infer_prompt.py \
    --config_name strat \
    --inputter_name strat_prompt \
    --add_nlg_eval false\
    --load_checkpoint /ziyuanqin/projects/nlp/comet/codes_zcj/DATA/strat_prompt.strat/2022-06-05060712.3e-05.16.1gpu/epoch-2.bin \
    --dqn_embed_checkpoint  /ziyuanqin/projects/nlp/comet/codes_zcj/DATA/stratrl.strat/2022-06-01152702.3e-05.16.1gpu/DQN_embed_636.bin \
    --dqn_checkpoint  /ziyuanqin/projects/nlp/comet/codes_zcj/DATA/stratrl.strat/2022-06-01152702.3e-05.16.1gpu/DQN_636.bin\
    --fp16 false \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 16 \
    --infer_input_file ./_reformat/test_anno_nocon.txt \
    --seed 13 \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 3
