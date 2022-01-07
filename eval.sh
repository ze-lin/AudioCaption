EXP_PATH=/home/zelin/xuenan/AudioCaption/experiments/audiocaps/pre_val/TransformerModel/panns_cnn14_rnn_attnrnn_origin1/seed_1
RAW_CSV_PATH=./data/audiocaps/test/lms.csv
FC_CSV_PATH=./data/audiocaps/test/panns_fc.csv
ATTN_CSV_PATH=./data/audiocaps/test/panns_attn.csv

CUDA_VISIBLE_DEVICES=3 python captioning/pytorch_runners/run.py evaluate $EXP_PATH \
    --task audiocaps \
    --save_type swa \
    --raw_feat_csv $RAW_CSV_PATH \
    --fc_feat_csv $FC_CSV_PATH \
    --attn_feat_csv  $ATTN_CSV_PATH \
    --caption_output swa.json \
    --score_output swa.txt \
    --method beam \
    --use_label False
