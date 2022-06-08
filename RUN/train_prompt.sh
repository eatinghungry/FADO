CUDA_VISIBLE_DEVICES=0 python train_prompt.py \
    --config_name strat \
    --inputter_name strat_prompt \
    --eval_input_file ./_reformat/dev_anno_nocon.txt \
    --seed 13 \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 3 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true