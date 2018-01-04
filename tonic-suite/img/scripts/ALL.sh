#!/usr/bin/env bash
declare -a RTT

for DIR in {1..12}
do
    mv ${DIR}_run 4_sem_results/
    for FILE in {25..64..1}
    do
        tail -n 900 ${FILE}.out >> temp
         OUTPUT="$(../${1}.py -i temp)"
         rm temp
         RTT[${FILE}]="${RTT[${FILE}]},${OUTPUT}"
    done
    cd ../
done

for i in {25..64}
do
    echo "${RTT[${i}]}"
done

