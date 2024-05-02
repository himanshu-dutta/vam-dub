#! /bin/bash

curl -X POST -F "files=@source_audio.wav" -F "files=@house.jpeg"  -o received_audio.wav http://10.195.100.4:9000/inference