#!/usr/bin/env bash

if [ "$EUID" -ne 0 ]
then
    echo "Please run as root"
    exit
fi

for CLOCK in {1..1..1}
do
    echo "power_avg,power_peak,clock_avg,clock_peak" > power_stats.out
    for RPS in {1..1..1}
    do
         ./djinn --common ../common/ --weights weights/ --portno 8080 --gpu 1 \
         --debug 0 --nets nets.txt --clock ${CLOCK}  --outfile ${RPS} \
         --threadcnt 2 
    done
    chown mchow:mchow *.out
    mkdir ${CLOCK}_run
    mv *.out ${CLOCK}_run

    chown mchow:mchow ${CLOCK}_run
done
