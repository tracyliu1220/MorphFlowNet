#! /bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da" -O checkpoint/FlowNet2_checkpoint.pth.tar && rm -rf /tmp/cookies.txt
