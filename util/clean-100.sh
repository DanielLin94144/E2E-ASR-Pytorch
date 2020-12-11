INPUT="/Home/daniel094144/End-to-End-jointCTC-Attention-ASR/corpus/clean100.txt"
OUTPUT="clean100-10k.model"
SIZE=10000
MODE="subword"

echo "__________generate subword model__________"
python generate_vocab_file.py --input ${INPUT} --output_file ${OUTPUT} --vocab_size ${SIZE} --mode ${MODE}