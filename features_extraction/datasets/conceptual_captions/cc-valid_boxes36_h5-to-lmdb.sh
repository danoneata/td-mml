#!/bin/bash

H5="/science/image/nlp-datasets/emanuele/data/conceptual_captions/features/cc-valid_boxes36.h5"
LMDB="/science/image/nlp-datasets/tt-mml/data/conceptual_captions/features/cc-valid_boxes36.lmdb"

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../..
python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB

deactivate

