#!/usr/bin/env bash
./sample_distribution.py 10
for CLOCK in {1..1..1}
do
    for RPS in {320..1..-1}
    do
        ./tonic-img --task imc --djinn 1  --input \
        imc-list.txt --hostname localhost --portno 8080
        sleep 7
    done
done
