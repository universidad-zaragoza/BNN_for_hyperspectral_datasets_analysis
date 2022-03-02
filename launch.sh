
# Normal training
./train.py 2>\dev\null | tee Logs/train_log.txt

# Mixed classes training
./train.py -m 2>\dev\null | tee Logs/mixed_train_log.txt

# Test
./test.py 2>\dev\null

# Noise test
./test_noise.py 2>\dev\null

# Mixed classes test
./test_mixed.py 2>\dev\null

