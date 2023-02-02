#!/bin/bash

# MACROS
TEST_EXECUTION=true

# Parameters
if [ "$TEST_EXECUTION" = true ]; then
    BO_epochs=100
    IP_epochs=100
    KSC_epochs=100
    PU_epochs=100
    SV_epochs=100
    BO_log=10
    IP_log=10
    KSC_log=10
    PU_log=10
    SV_log=10
else
    BO_epochs=17000
    IP_epochs=22000
    KSC_epochs=41000
    PU_epochs=1800
    SV_epochs=4000
    BO_log=1000
    IP_log=1000
    KSC_log=1000
    PU_log=100
    SV_log=100
fi

# Make Logs and Test dirs if they don't exist
mkdir -p Logs
mkdir -p Test

# Normal training
./train.py BO $BO_epochs $BO_log 2>\dev\null | tee Logs/BO_train_log.txt
./train.py IP $IP_epochs $IP_log 2>\dev\null | tee Logs/IP_train_log.txt
./train.py KSC $KSC_epochs $KSC_log 2>\dev\null | tee Logs/KSC_train_log.txt
./train.py PU $PU_epochs $PU_log 2>\dev\null | tee Logs/PU_train_log.txt
./train.py SV $SV_epochs $SV_log 2>\dev\null | tee Logs/SV_train_log.txt

# Test
./test.py $BO_epochs $IP_epochs $KSC_epochs $PU_epochs $SV_epochs 2>\dev\null

# Map test
./test_map.py BO $BO_epochs 2>\dev\null
./test_map.py IP $IP_epochs 2>\dev\null
./test_map.py KSC $KSC_epochs 2>\dev\null
./test_map.py PU $PU_epochs 2>\dev\null
./test_map.py SV $SV_epochs 2>\dev\null

# Test noisy data
./test_noise.py $BO_epochs $IP_epochs $KSC_epochs $PU_epochs $SV_epochs 2>\dev\null

# Mixed classes training
./train.py BO $BO_epochs $BO_log -m 2>\dev\null | tee Logs/BO_mixed_train_log.txt
./train.py IP $IP_epochs $IP_log -m 2>\dev\null | tee Logs/IP_mixed_train_log.txt
./train.py KSC $KSC_epochs $KSC_log -m 2>\dev\null | tee Logs/KSC_mixed_train_log.txt
./train.py PU $PU_epochs $PU_log -m 2>\dev\null | tee Logs/PU_mixed_train_log.txt
./train.py SV $SV_epochs $SV_log -m 2>\dev\null | tee Logs/SV_mixed_train_log.txt

# Mixed classes test
./test_mixed.py $BO_epochs $IP_epochs $KSC_epochs $PU_epochs $SV_epochs 2>\dev\null

