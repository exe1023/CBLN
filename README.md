## CBLN
Pytorch 0.3.0 implementation of Conditioned Batch Normalization / Conditioned Layer Normalization

### Data
To download the VQA dataset please use the script 'scripts/vqa_download.sh': </br>
```
scripts/vqa_download.sh `pwd`/data
```

### Process Data
Detailed instructions for processing data are provided by [GuessWhatGame/vqa](https://github.com/GuessWhatGame/vqa#introduction). </br>

#### Create dictionary
To create the VQA dictionary, use the script preprocess_data/create_dico.py. </br>
```
python3 create_dictionary.py --data_dir data --year 2014 --dict_file dict.json
```

#### Create GLOVE dictionary
To create the GLOVE dictionary, download the original glove file and run the script preprocess_data/create_gloves.py. </br>
```
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/
unzip data/glove.42B.300d.zip -d data/
python3 create_gloves.py --data_dir data --glove_in data/glove.42B.300d.txt --glove_out data/glove_dict.pkl --year 2014
```

### Train Model
To train the network, set the required parameters in ``` config.json ``` and run the script main.py.
```
bash train.sh
```

