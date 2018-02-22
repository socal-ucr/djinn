#!/usr/bin/env bash
for CLOCK in {1..1..1}
do
    for RPS in {10..10..1}
    do
        OUTFILE=${RPS}
        ./sample_distribution.py ${RPS}
        ./tonic-img --task imc --djinn 1  --input \
        imc-list.txt --hostname localhost --portno 8080
        sleep 5
    done
 #   mkdir ${CLOCK}_run
  #  mv *.out ${CLOCK}_run
done
