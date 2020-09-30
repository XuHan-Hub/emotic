#!/bin/bash

# Image and model names
MODEL_NAME=ade20k-resnet101dilated-ppm_deepsup
MODEL_PATH=ckpt/$MODEL_NAME

ENCODER=$MODEL_NAME/encoder_epoch_25.pth
DECODER=$MODEL_NAME/decoder_epoch_25.pth

for DOC_NAME in ade20k emodb_small framesdb mscoco
  do
  # Inference
  python3 -u test.py \
    --imgs ../proj/data/emotic19/emotic/$DOC_NAME/images \
    --gpu 1 \
    --cfg config/ade20k-resnet101dilated-ppm_deepsup.yaml \
    DIR $MODEL_PATH \
    TEST.result ../proj/data/emotic19/emotic_seg/$DOC_NAME/images \
    TEST.checkpoint epoch_25.pth
  done

