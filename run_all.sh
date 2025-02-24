# run batch steganography encoding (just encryption plus encoding steps)
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt utf8 -encode bins -block_size 3 -device 0
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt cached -encode huffman -block_size 7 -device 0
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt utf8 -encode arithmetic -topK 300 -device 0
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt utf8 -encode saac -delta 0.01 -device 0
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt utf8 -encode saac -delta 0.05 -device 0
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt utf8 -encode saac -delta 0.1 -device 0
# run single steganography whole pipeline (encryption -- encoding -- decoding -- decryption)
python run_single_end2end.py -encode bins -device 0
python run_single_end2end.py -encode huffman -device 0
python run_single_end2end.py -encode arithmetic -device 0
python run_single_end2end.py -encode saac -device 0
python run_single_end2end.py -encode dynamic -device 0
python main.py

nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode saac -delta 0.01 -device 0 > covid_saac_4_delta001.log 2>&1 &
nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode saac -delta 0.05 -device 0 > covid_saac_4_delta005.log 2>&1 &


echo "cd /home/ubuntu/stega && " | at now + 4 hours


echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset cnn_dm -dataset_path ./datasets/cnn_dm -encrypt utf8 -encode bins -block_size 3 -device 0 > news_block_3.log 2>&1" | at now + 2 hours
nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode bins -block_size 3 -device 0 > covid_block_3.log 2>&1 &
echo "cd /home/ubuntu/stega/reject && nohup python main.py > reject6drug450covid450news.log 2>&1 &" | at now + 3 hours

# nohup python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt utf8 -encode arithmetic -topK 900 -device 0 > drug_ari_k900.log 2>&1 &  arith方法drug结束
nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode arithmetic -topK 300 -device 0 > covid_ari_k300.log 2>&1 &

echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode arithmetic -topK 600 -device 0 > covid_ari_k600.log 2>&1" | at now + 1 hours
echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode arithmetic -topK 900 -device 0 > covid_ari_k900.log 2>&1" | at now + 1 hours
echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset cnn_dm -dataset_path ./datasets/cnn_dm -encrypt utf8 -encode arithmetic -topK 300 -device 0 > news_ari_k300.log 2>&1" | at now + 2 hours
echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset cnn_dm -dataset_path ./datasets/cnn_dm -encrypt utf8 -encode arithmetic -topK 600 -device 0 > news_ari_k600.log 2>&1" | at now + 3 hours
echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset cnn_dm -dataset_path ./datasets/cnn_dm -encrypt utf8 -encode arithmetic -topK 900 -device 0 > news_ari_k900.log 2>&1" | at now + 4 hours
echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode bins -block_size 3 -device 0 > covid_block3.log 2>&1" | at now + 5 hours
echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset cnn_dm -dataset_path ./datasets/cnn_dm -encrypt utf8 -encode bins -block_size 3 -device 0 > news_block3.log 2>&1" | at now + 6 hours

# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode bins -block_size 3 -device 0 > random_block3.log 2>&1 &" | at now + 7 hours
# 已定时
# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode saac -delta 0.1 -device 0 > random_saac01.log 2>&1" | at now + 2 hours
# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode saac -delta 0.05 -device 0 > random_saac005.log 2>&1" | at now + 3 hours

# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode saac -delta 0.01 -device 0 > random_saac001.log 2>&1" | at now + 1 hours
# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode arithmetic -topK 300 -device 0 > random_ari_k300.log 2>&1" | at now + 2 hours
# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode arithmetic -topK 600 -device 0 > random_ari_k600.log 2>&1" | at now + 3 hours
# echo "cd /home/ubuntu/stega && nohup python run_batch_encode.py -dataset random -dataset_path ./datasets/random -encrypt cached -encode arithmetic -topK 900 -device 0 > random_ari_k900.log 2>&1" | at now + 3 hours
echo "cd /home/ubuntu/stega/reject && nohup python -u randompath.py > random6reject.log 2>&1" | at now + 4 hours
echo "cd /home/ubuntu/stega/reject && nohup python -u randompath2.py > random8reject.log 2>&1" | at now + 6 hours
# echo "cd /home/ubuntu/stega/expr && nohup python -u randompath.py > random4static.log 2>&1" | at now + 7 hours

# nohup python run_batch_encode.py -dataset covid_19 -dataset_path ./datasets/covid_19 -encrypt utf8 -encode bins -block_size 3 -device 0 > drug_block3.log 2>&1 & 本地已跑
