sh cuda_mem.sh
python main.py --name "pydsje" \
  --data-dir "../StackGAN-Pytorch/data/sixray_2381" \
  --class-name-file "../StackGAN-Pytorch/data/sixray_2381/train/classes.txt" \
  --label-csv "../StackGAN-Pytorch/data/sixray_2381/train/labels.csv" \
  --drop-last \
  --emd-dim 1536 \
  --batch-size 128 \
  --epoch-start 0 \
  --max-epoch 1000 \
  --embedding-strategy char \
  --doc-length 350 \
  --cuda
#  --finetune
#  --epoch-start 501 \
#  --max-epoch 1000 \
#  --finetune \
#  --NET-IMG 'output/pydsje_word_train_2023_07_05_18_32_24/Model/netIMG_epoch_500.pth' \
#  --NET-TXT 'output/pydsje_word_train_2023_07_05_18_32_24/Model/netTXT_epoch_500.pth' \
