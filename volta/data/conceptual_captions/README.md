# Conceptual Captions

We provide the lists of [training](train_ids.txt) and [validation](valid_ids.txt) images used in our study.
Each image URL is [encoded](preprocess_cc_train.py) as follows:
```python
def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))
```

## Download 
Run `download_data.py` from the directory in which you want to store your corpus.

For more details, 
check out the [ViLBERT repository](https://github.com/jiasenlu/vilbert_beta/tree/master/tools/DownloadConcptualCaption).

## Extract Image Features
From the `volta` repository root, first use `airsplay/bottom-up-attention` to extract image features with Faster R-CNN. 
For example:
```text
# Path to training images
IMG_DIR=/home/bugliarello.e/data/conceptual_captions/images/training

docker run \
    --gpus all \
    -v $IMG_DIR:/workspace/images:ro \
    -v $(pwd)/data/conceptual_captions:/workspace/features \
    -v $(pwd)/data/snap:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd features/
CUDA_VISIBLE_DEVICES=0 python extract_cc_image.py --split train --group_id 0 --total_group 8
```
In this example, as we have 8 groups, the code above will need to be run 8 times, with `group_id` ranging from 0 to 7. 

NB. This script will extract features in this folder.
If you want to extract them in a different directory, make a copy of these two files in that directory,
and change `$(pwd)/data/conceptual_captions` in the docker execution appropriately.


Then, merge all the features extracted from the different processes:
```text
DATA_DIR=/home/bugliarello.e/data/conceptual_captions/imgfeats

python merge_cc_image.py --split train --total_group 8 --datadir $DATA_DIR
```

## Serialize Image Features & Extract Captions
Finally, serialize the image features and extract their captions:
```text
bash preprocess_cc_train.sh
```

---

The corpus directory looks as follows:
```text
conceptual_captions/
 |-- annotations/
 |    |-- caption_train.json
 |    |-- caption_valid.json
 |
 |-- images/
 |    |-- training/
 |    |-- validation/
 |
 |-- resnet101_faster_rcnn_genome_imgfeats/
 |    |-- train_obj36-36.tsv
 |    |-- valid_obj36-36.tsv
 |    |-- volta/
 |    |    |-- training_feat_all.lmdb
 |    |    |-- validation_feat_all.lmdb
 |
 |-- Train_GCC-training.tsv
 |-- Validation_GCC-1.1.0-Validation.tsv

```

## Translate Captions
Translate the English CC captions into a target language using the M2M-100 translation model:
```bash
python translate_cc.py -s train -m m2m-100-md -l ro
```
- For the `train` split, the script translates only a subset of the captions (proportional to the amount of data used for training the XML-R model; as specified by `data/xmlr-counts.csv`)
- For the `valid` split, the script translates the entire validation data

Translate the entire English CC captions:
```bash
python translate_cc_full.py -s train -m m2m-100-md -l ro
```

---

## Translate all captions in the IGLUE langauges using the large translation model

Download the keys for which we are lacking a translation
```bash
wget https://sharing.speed.pub.ro/owncloud/index.php/s/k9HEyTUy0du8aTO -O data/missing-keys.json
```

Install Python requirements:
```bash
pip install -r requirements.txt
```

Run translation:
```bash
python translate_cc_full.py --s train -l ar -m m2m-100-lg -b 8 -k data/missing-keys.json --device cuda:0
```

The `-b` flag sets the batch size.
For the large translation model, `m2m-100-lg` I have used a batch size of 8 on machines with 15GB of memory (Tesla T4).
