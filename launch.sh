
# Normal training
./train.py 2>\dev\null | tee Logs/train_log.txt

# Mixed classes training
./train.py -m 2>\dev\null | tee Logs/mixed_train_log.txt

# Test
./test.py 2>\dev\null | tee Logs/test_log.txt

# Mixed classes test
./test_mixed.py 2>\dev\null | tee Logs/mixed_test_log.txt

