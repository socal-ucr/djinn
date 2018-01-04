#!/usr/bin/env bash
for RUN in {01..04..1}
do
    PORT=80${RUN}
    ./tonic-img --task imc --rps ${RUN} --djinn 1 --seconds 90 --input imc-list.txt --hostname 192.168.1.170 --portno ${PORT} &
done
