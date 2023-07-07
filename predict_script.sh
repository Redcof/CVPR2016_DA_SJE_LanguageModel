sh cuda_mem.sh
python main.py --name "jointemb" \
--data-dir "../StackGAN-Pytorch/data/sixray_2381" \
--class-name-file "../StackGAN-Pytorch/data/sixray_2381/train/classes.txt" \
--label-csv "../StackGAN-Pytorch/data/sixray_2381/train/labels.csv" \
--NET-TXT "output/pydsje_word_train_2023_07_07_13_29_32/Model/netTXT_epoch_230.pth" \
--vocabulary-txt "output/pydsje_word_train_2023_07_07_13_29_32/Model/vocabulary.txt" \
--drop-last \
--emd-dim 1536 \
--batch-size 128 \
--doc-length 70 \
--embedding-strategy word \
--predict \
--bulk


