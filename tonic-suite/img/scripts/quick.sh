#!/usr/bin/env bash

TESTNAME=one_socket

#for DIR in {1..18}
#do
 #   cd ${DIR}_run
    for FILE in {10..32..1}
    do
        # OUTPUT="$(../${1}.py -i ${FILE}.out)"
        tail -n 900 ${FILE}_${TESTNAME}.out >> temp
         ./${1}.py -i temp
        # RTT[${FILE}]="${RTT[${FILE}]},${OUTPUT}"
        rm temp
    done
  #  cd ../
#done

#for i in {10..64}
#do
    #echo "${RTT[${i}]}"
#    echo "${i}"

#done
