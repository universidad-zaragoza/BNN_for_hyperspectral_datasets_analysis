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

# # Normal training
# ./train.py BO $BO_epochs $BO_log 2>\dev\null | tee Logs/BO_train_log.txt
# ./train.py IP $IP_epochs $IP_log 2>\dev\null | tee Logs/IP_train_log.txt
# ./train.py KSC $KSC_epochs $KSC_log 2>\dev\null | tee Logs/KSC_train_log.txt
# ./train.py PU $PU_epochs $PU_log 2>\dev\null | tee Logs/PU_train_log.txt
# ./train.py SV $SV_epochs $SV_log 2>\dev\null | tee Logs/SV_train_log.txt
# 
# # Test
# ./test.py $BO_epochs $IP_epochs $KSC_epochs $PU_epochs $SV_epochs 2>\dev\null
# 
# # Map test
# ./test_map.py BO $BO_epochs 2>\dev\null
# ./test_map.py IP $IP_epochs 2>\dev\null
# ./test_map.py KSC $KSC_epochs 2>\dev\null
# ./test_map.py PU $PU_epochs 2>\dev\null
# ./test_map.py SV $SV_epochs 2>\dev\null
# 
# # Combine maps
# cd Test
# convert -density 600 BO_RGB.pdf BO_gt.pdf BO_${BO_epochs}_pred_map.pdf BO_${BO_epochs}_H_map.pdf +append H_BO.pdf
# convert -density 500 IP_RGB.pdf IP_gt.pdf +append H_IP_top.pdf
# convert -density 500 IP_${IP_epochs}_pred_map.pdf IP_${IP_epochs}_H_map.pdf +append H_IP_bottom.pdf
# convert H_IP_top.pdf H_IP_bottom.pdf -append H_IP.pdf
# rm H_IP_top.pdf
# rm H_IP_bottom.pdf
# convert -density 350 KSC_RGB.pdf KSC_gt.pdf +append H_KSC_top.pdf
# convert -density 350 KSC_${KSC_epochs}_pred_map.pdf KSC_${KSC_epochs}_H_map.pdf +append H_KSC_bottom.pdf
# convert H_KSC_top.pdf H_KSC_bottom.pdf -append H_KSC.pdf
# rm H_KSC_top.pdf
# rm H_KSC_bottom.pdf
# convert -density 600 PU_RGB.pdf PU_gt.pdf PU_${PU_epochs}_pred_map.pdf PU_${PU_epochs}_H_map.pdf +append H_PU.pdf
# convert -density 600 SV_RGB.pdf SV_gt.pdf SV_${SV_epochs}_pred_map.pdf SV_${SV_epochs}_H_map.pdf +append H_SV.pdf
# cd ..
# 
# # Test noisy data
# ./test_noise.py $BO_epochs $IP_epochs $KSC_epochs $PU_epochs $SV_epochs 2>\dev\null

# Mixed classes training
# ./train.py BO $BO_epochs $BO_log -m 2>\dev\null | tee Logs/BO_mixed_train_log.txt
# ./train.py IP $IP_epochs $IP_log -m 2>\dev\null | tee Logs/IP_mixed_train_log.txt
# ./train.py KSC $KSC_epochs $KSC_log -m 2>\dev\null | tee Logs/KSC_mixed_train_log.txt
# ./train.py PU $PU_epochs $PU_log -m 2>\dev\null | tee Logs/PU_mixed_train_log.txt
# ./train.py SV $SV_epochs $SV_log -m 2>\dev\null | tee Logs/SV_mixed_train_log.txt

# Mixed classes test
./test_mixed.py $BO_epochs $IP_epochs $KSC_epochs $PU_epochs $SV_epochs 2>\dev\null

