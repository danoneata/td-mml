#!/bin/bash

DATA="/science/image/nlp-datasets/emanuele/data/conceptual_captions"
TSV="${DATA}/resnet101_faster_rcnn_genome_imgfeats/valid_obj36-36.clean.tsv"
H5="${DATA}/features/cc-valid_boxes36.h5"

source activate /science/image/nlp-datasets/emanuele/envs/volta

cd ../..
python tsv_to_h5.py --tsv $TSV --h5 $H5

conda deactivate

