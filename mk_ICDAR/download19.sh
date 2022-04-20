#!/usr/bin/env bash

for url in $(cat urls19.txt | tr -d '\r')
do
    wget $url --no-check-certificate
done
for i in *.zip
do
  unzip $i -d /opt/ml/input/data/ICDAR19_MLT/raw/training_images
done
wget http://datasets.cvc.uab.es/rrc/train_gt_t13.zip --no-check-certificate
unzip train_gt_t13.zip -d /opt/ml/input/data/ICDAR19_MLT/raw/training_gt