sh cuda_mem.sh
python main.py --name "jointemb955" \
--data-dir "../StackGAN-Pytorch/data/sixray_2381/train/captions" \
--class-name-file "../StackGAN-Pytorch/data/sixray_2381/train/classes.txt" \
--label-csv "../StackGAN-Pytorch/data/sixray_2381/train/labels.csv" \
--NET-TXT "output/pydsje_word_train_2023_07_08_11_32_47/Model/netTXT_epoch_955.pth" \
--vocabulary-txt "output/pydsje_word_train_2023_07_08_11_32_47/Log/vocabulary.txt" \
--drop-last \
--emd-dim 1536 \
--batch-size 128 \
--doc-length 70 \
--embedding-strategy word \
--predict \
--bulk 4 \
--cuda


