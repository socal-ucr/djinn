#!/usr/bin/env bash
for CLOCK in {2..2..1}
do
    mkdir ${CLOCK}_run
    chown wong936:wong936 ${CLOCK}_run
    for RPS in {320..1..-1}
    do
         ./djinn --common ../common/ --weights weights/ --portno 8080 --gpu 0 \
         --debug 0 --nets nets.txt --clock -1  --outfile ${RPS} \
         --tbrf ${RPS} --threadcnt 1 
        chown wong936:wong936 *.out
        mv *.out ${CLOCK}_run
    done
done
