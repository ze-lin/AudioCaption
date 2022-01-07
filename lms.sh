OUTPUT_PATH=data/audiocaps

for SPLIT in train val test; 
  do python data/extract_feature.py $OUTPUT_PATH/$SPLIT/wav.csv $OUTPUT_PATH/$SPLIT/lms.h5 $OUTPUT_PATH/$SPLIT/lms.csv lms -win_length 640 -hop_length 320 -n_mels 64; 
done