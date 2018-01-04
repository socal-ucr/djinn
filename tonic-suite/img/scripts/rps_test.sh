#!/usr/bin/env bash
for RUN in {1..16..1}
do
        ./tonic-img --task imc --rps ${RUN} --djinn 1 --seconds 90 --input imc-list.txt --hostname 192.168.1.170 --portno 8080 &
done
