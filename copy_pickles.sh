# ####################################
# sh copy_pickles.sh "jointemb995_word_predict_2023_07_10_16_32_43" "../StackGAN-Pytorch/data/sixray_2381"
rm "$2/train/filenames.pickle"
rm "$2/train/embedding_bulk_word_1536_jemb.pickle"
rm "$2/test/embedding_bulk_word_1536_jemb.pickle"
cp "output/$1/Image/filenames_bulk_jemb.pickle" "$2/train/filenames.pickle"
cp "output/$1/Image/embedding_bulk_word_1536_jemb.pickle" "$2/train/embedding_bulk_word_1536_jemb.pickle"
cp "output/$1/Image/embedding_test_bulk_word_1536_jemb.pickle" "$2/test/embedding_bulk_word_1536_jemb.pickle"