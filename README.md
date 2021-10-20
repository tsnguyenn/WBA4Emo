# Window-based Attention model for Emotion Understanding

This repo contains the implementation of the window-based model presented in the journal paper ["Attention uncovers task-relevant semantics in emotional narrative understanding"](https://www.sciencedirect.com/science/article/abs/pii/S0950705121004251) (Knowledge-Based Systems - Volume 226, 17 August 2021, 107162)

## Preprocessing 
1. Download the Stanford Emotional Narratives Dataset (SEND) [here](https://github.com/StanfordSocialNeuroscienceLab/SEND). We will use the linguistic data. 
2. Convert the dataset into window-based data:
```
python3 src/preprocessing/prepare_windows.py --input SENDv1_featuresRatings_pw/features 
--ratings SENDv1_featuresRatings_pw/ratings --window 5 --output data/preprocessed/window-based_5s/ 
```
where ```--input```, ```--ratings``` are the paths to the 'features' and 'ratings' in SEND, respectively. ```--window``` is the window size (in second), and ```--output``` is where you want to save the window-based data. 

3. Precompute word embeddings. 
* GloVe: unzip the filtered GloVe embeddings file ([data/preprocessed/embeddings/glove.840B.300d.word2vec.filtered.zip](https://github.com/jsonnguyen/WBA4Emo/blob/master/data/preprocessed/embeddings/glove.840B.300d.word2vec.filtered.zip))
* Other pre-trained models: 
```
python3 src/preprocessing/precompute_embeddings.py --model bert --gpu 3 
--input data/preprocessed/window-based_5s --output data/preprocessed/embeddings/
```
where ```--model``` is the pretrained model you want to use. Implemented models: 'roberta', 'gpt2', 'electra', 'bert', 'distill', 'transformerxl'. ```--input``` is the window-based data preprocessed in (2), ```--output``` where you want to store the pre-computed embeddings to be used later (the actual dir will be automatically created). 

5. 
6. 

The preprocessed files are included to run the models. The original dataset can be found [here - SEND](https://github.com/StanfordSocialNeuroscienceLab/SEND)

## Run Window-based with Attention (WBA)
This is to train/test window-based **with attention** (*WBA*) models. 

Before running the commands, you should double check (and modify if needed) the directory arguments:
* --data: path to folder 'words_window-based_**n**\_seconds/' where **n** is the window size (i.e., **n** = {3, 5, 10})
* --split_file: path to file 'splits.json'
* --gt_dir: path to folder 'observers_EWE/'
* --embeddings: path to embedding file 'glove.840B.300d.word2vec.filtered'

Other arguments: 
* --attention: 
  * 'yes': use attention (WBA)
  * 'no': don't use attention (window-based **without** attention, WB) 
* --runs: number of times training with different seeds 
* --epoch: number of epochs to train in each run. 
* --batch: batch size 
* --lr: (initial) learning rate
* --h1: local-level LSTM hidden size 
* --h2: global-level LSTM hidden size
* --event_len: padding len 
* --bi: whether to use bidirectional LSTM 
* --optimizer: sgd or adam 

Commands below are for window size **n=5**

***Training:*** 
```
python3 src/models/WBA.py --mode train --attention yes --data data/window_based_data/words_window-based_5_seconds/ --runs 20  --gpu 0 --embeddings data/glove.840B.300d.word2vec.filtered --splits data/splits.json --epoch 200 --batch 117 --lr 0.001 --h1 128 --h2 128 --optimizer adam --dropout 0.01 --event_min 2 --event_len 25 --suffix WBA5 --bi yes 
```

***Testing (a pretrained model):***
* --mode test
* --saved_model: path to the saved model 
* --output_dir: path to the output dir to store outputs (the folder should be already created)

```
python3 src/models/WBA.py --mode test --saved_model data/examples/saved_model/wba-5s.pt --output_dir data/examples/output_tmp/wba-5s --attention yes --data data/window_based_data/words_window-based_5_seconds/  --gpu 0 --embeddings data/glove.840B.300d.word2vec.filtered --splits data/splits.json --h1 128 --h2 128 --event_min 2 --event_len 25 --bi yes 
```


***Testing (after training):***
* Change '--mode' to 'test', 
* add '--id' (program id: the folder name) 
* and '--name': model file's name
```
python3 src/models/WBA.py --mode test --id 20200526_154625_WBA5 --name run_1_best_116.pt --attention yes --data data/window_based_data/words_window-based_5_seconds/ --runs 20  --gpu 0 --embeddings data/glove.840B.300d.word2vec.filtered --splits data/splits.json --epoch 200 --batch 117 --lr 0.001 --h1 128 --h2 128 --optimizer adam --dropout 0.01 --event_min 2 --event_len 25 --suffix WBA5 --bi yes 
```

## Run baselines 
### Window-based without Attention (WB)
Exactly the same as in **WBA**, but '--attention' is set to 'no', i.e. '--attention no'


### Flattened-LSTM (F-LSTM) 
This part is to re-produce the baseline, F-LSTM. 

Before running the commands, you should double check (and modify if needed) the directory arguments:
* --data: path to folder 'EWE_all_words_withScores'
* --split_file: path to file 'splits.json'
* --gt_dir: path to folder 'observers_EWE/'
* --embeddings: path to embedding file 'glove.840B.300d.word2vec.filtered'

***Training:***
```
python3 src/models/F_LSTM.py --mode train --batch 117 --epoch 200 --hidden 128 --runs 20 --lr 0.001 --dropout 0.01 --gpu 0 --bi yes --data data/EWE_all_words_withScores --split_file data/splits.json --gt_dir data/observers_EWE/ --embeddings data/glove.840B.300d.word2vec.filtered --suffix F-LSTM
```

***Testing (a pretrained model):***
* --mode test 
* --saved_model: path to the saved model 
* --output_dir: path to the output dir to store outputs (the folder should be already created)
```
python3 src/models/F_LSTM.py --mode test  --saved_model data/examples/saved_model/f-lstm.pt --output_dir data/examples/output_tmp/f-lstm --hidden 128 --gpu 0 --bi yes --data data/EWE_all_words_withScores --split_file data/splits.json --gt_dir data/observers_EWE/ --embeddings data/glove.840B.300d.word2vec.filtered 
```

***Testing (after training):***
* Change '--mode' to 'test', 
* add '--id' (program id: the folder name) 
* and '--name': model file's name
```
python3 src/models/F_LSTM.py --mode test --id 20200526_122326_F-LSTM --name run_13_best_144.pt --batch 117 --epoch 200 --hidden 128 --runs 20 --lr 0.001 --dropout 0.01 --gpu 0 --bi yes --data data/EWE_all_words_withScores --split_file data/splits.json --gt_dir data/observers_EWE/ --embeddings data/glove.840B.300d.word2vec.filtered --suffix F-LSTM 
```

## Run Analysis
### Word Cloud 

To generate Word Cloud for TEST set: 
```
python3 src/analysis/word_cloud.py --dataset test --split_file data/splits.json --attention_dir data/examples/output_tmp/wba-5s/attention_output --weights_output_file data/examples/output_tmp/wba-5s/attention_weights_word.json --wordcloud_output_file data/examples/output_tmp/wba-5s/attention_weights_word.png  
```
WordCloud for Test set (WBA-5s) 

<img src="./images/wordcloud_wba5_all.png" width="500" align='center'>

### Heatmap 
To create heatmap (i.e., highlighted text based on attention weights), you can use the following command: 

```
python3 src/analysis/heatmap.py --input data/examples/output_tmp/wba-5s/attention_output/ID112_vid2.txt
```

This will generate latex table, the content is copied to clipboard. You can paste to latex file to view the results. 

Example of output: 
![Heatmap example](./images/heatmap.png)
