python prepare_dqn.py \
    --config_name strat \
    --inputter_name stratdqn \
    --train_input_file ./_reformat/train_anno_nocon.txt \
    --max_input_length 160 \
    --max_decoder_input_length 40 \