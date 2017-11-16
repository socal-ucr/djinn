#!/usr/bin/env bash
for CLOCK in {1..18..1}
do
    for RPS in {10..64..1}
    do
        sudo ./djinn --common ../common/ --weights weights/ --portno 8080 --gpu 1 --debug 0 --nets nets.txt --clock ${CLOCK}  --outfile ${RPS} --threadcnt 1000
    done
    mkdir ${CLOCK}_run
    mv *.out ${CLOCK}_run/.
done
