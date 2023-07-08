sh cuda_mem.sh
python main.py --name "jointemb" \
--data-dir "../StackGAN-Pytorch/data/sixray_2381/train/captions" \
--class-name-file "../StackGAN-Pytorch/data/sixray_2381/train/classes.txt" \
--label-csv "../StackGAN-Pytorch/data/sixray_2381/train/labels.csv" \
--NET-TXT "output/pydsje_train_2023_06_30_15_23_04/Model/netTXT_epoch_175.pth" \
--drop-last \
--emd-dim 1536 \
--batch-size 128 \
--doc-length 350 \
--embedding-strategy char \
--predict \
--bulk 4


