sh cuda_mem.sh
python main.py --name "pydsje" \
  --data-dir "../StackGAN-Pytorch/data/sixray_2381" \
  --class-name-file "../StackGAN-Pytorch/data/sixray_2381/train/classes.txt" \
  --label-csv "../StackGAN-Pytorch/data/sixray_2381/train/labels.csv" \
  --drop-last \
  --emd-dim 1536 \
  --batch-size 128 \
  --max-epoch 180 \
  --embedding-strategy char \
  --doc-length 170 \
  --cuda
