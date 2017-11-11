#!/usr/bin/env bash
sudo ./djinn --common ../common/ --weights weights/ --portno 8080 --gpu 1 --debug 0 --nets nets.txt
