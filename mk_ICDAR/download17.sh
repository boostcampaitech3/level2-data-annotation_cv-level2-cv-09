#!/usr/bin/env bash

for url in $(cat urls17.txt | tr -d '\r')
do
    wget $url --no-check-certificate
done
for i in *.zip
do
  unzip $i -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_training_images
done
wget https://datasets.cvc.uab.es/rrc/ch8_training_localization_transcription_gt_v2.zip --no-check-certificate
unzip ch8_training_localization_transcription_gt_v2.zip -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_training_gt
wget https://rrc.cvc.uab.es/downloads/ch8_validation_images.zip --no-check-certificate
unzip ch8_validation_images.zip -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_validation_images
wget https://datasets.cvc.uab.es/rrc/ch8_validation_localization_transcription_gt_v2.zip --no-check-certificate
unzip ch8_validation_localization_transcription_gt_v2.zip -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_validation_gt