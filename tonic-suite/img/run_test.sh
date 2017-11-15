#!/usr/bin/env bash
for CLOCK in {1..18..1}
do
    for RPS in {10..64..1}
    do
        ./sample_distribution.py ${RPS}
        ./tonic-img --task imc --rps ${RPS} --djinn 1 --seconds 90 --input imc-list.txt --hostname 192.168.1.170 --portno 8080
        sleep 5
    done
    mkdir ${CLOCK}_run
    mv *.out ${CLOCK}_run
done
