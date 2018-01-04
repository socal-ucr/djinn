#!/usr/bin/env bash
for CLOCK in {1..1..1}
do
    for RPS in {10..32..1}
    do
        OUTFILE=${RPS}_one_socket
        ./sample_distribution.py ${RPS}
        ./tonic-img --task imc --outfile ${OUTFILE} --djinn 1 --seconds 200 --input imc-list.txt --hostname 192.168.1.170 --portno 8081
        sleep 5
    done
 #   mkdir ${CLOCK}_run
  #  mv *.out ${CLOCK}_run
done
