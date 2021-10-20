# Window-based Attention model for Emotion Understanding

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
