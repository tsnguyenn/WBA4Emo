# Window-based Attention (WBA) model for Emotion Understanding

This repo contains the implementation of the window-based model presented in the journal paper "[Attention uncovers task-relevant semantics in emotional narrative understanding](https://www.sciencedirect.com/science/article/abs/pii/S0950705121004251)" \[[pdf](https://www.dropbox.com/s/mhmbaq1meeb2lva/Narrative-WBA-KB21.pdf?dl=0)\] (Knowledge-Based Systems - Volume 226, 17 August 2021, 107162)

The proposed Window-based Attention (WBA) model includes a hierarchical (two-level) LSTM with attention mechanism to predict real-valued emotion valence in time-series: 
![WBA Model for Emotion Valence Prediction](https://github.com/tsnguyen-pk/WBA4Emo/blob/master/data/img/WBA_model.jpg)


## Preprocessing 
**1. Download the Stanford Emotional Narratives Dataset (SEND)** [here](https://github.com/StanfordSocialNeuroscienceLab/SEND). We will use the linguistic data. 

**2. Convert the dataset into window-based data**:
```
python3 src/preprocessing/prepare_windows.py --input SENDv1_featuresRatings_pw/features 
--ratings SENDv1_featuresRatings_pw/ratings --window 5 --output data/preprocessed/window-based_5s/ 
```
where ```--input```, ```--ratings``` are the paths to the 'features' and 'ratings' in SEND, respectively. ```--window``` is the window size (in second), and ```--output``` is where you want to save the window-based data. 

Note that files in 'features' are for word level and should contain the following information in the first 3 columns: _time-onset_, _time-offset_, and _word_. 


**3. Precompute word embeddings.** 
```
python3 src/preprocessing/precompute_embeddings.py --model bert --gpu 3 
--input data/preprocessed/window-based_5s --output data/preprocessed/embeddings/
```
where 
* ```--model``` is the pretrained model you want to use. Available models: 'roberta', 'gpt2', 'electra', 'bert', 'distill', 'transformerxl'. 
* ```--input``` is the window-based data preprocessed in (2), 
* ```--output``` where you want to store the pre-computed embeddings to be used later (the actual dir will be automatically created). 

## Training and testing the WBA model
**1. Training**

Results reported in the paper is the average of the best epochs when running the training process for 20 times. In each run, we chose the best epoch based on the performance in the validation set ('valid'). You can train the model using the following command: 
```
python3 src/models/wba_model.py --data data/preprocessed/window-based_5s --ratings data/ratings/ 
--embeddings data/preprocessed/embeddings/window-based_5s_bert --mode train --epoch 200 --batch 117 --lr 0.001 --h1 128 --h2 128 --runs 20 
--optimizer adam --dropout 0.01 --gpu 3 --attention yes
```
where 
* ```--data``` is the path to the preprocessed window-based data 
* ```--ratings``` is the path to the 'ratings' folder in SEND
* ```--embeddings``` is the path to the precomputed embeddings
* ```--mode```: we are in 'train' mode (available modes: train, test)
* ```--attention```: whether to use attention
* ```--runs```: the number of training times you want to run. 
* others: use ```--help``` to see other arguments. 

You might see error about importing local library. If so, you can add your local project using the following command: 
```
sys.path.append("/path/to/your/project")
```

This version currently supports running 'roberta', 'gpt2', 'electra', 'bert', 'distill', 'transformerxl'

**2. Testing** 

Once trained a model, you can test a particular model using the following command. 

```
python3 src/models/wba_model.py --embeddings data/preprocessed/embeddings/window-based_5s_bert 
--data data/preprocessed/window-based_5s --ratings data/ratings/ 
--epoch 200 --batch 117 --lr 0.001 --h1 128 --h2 128 --runs 20 --optimizer adam --dropout 0.01 --gpu 3 --attention yes 
--mode test --model data/log/20211020_161126_WB_5_bert+our_bilstm/models/run_0_best_83.pt 
--output data/log/20211020_161126_WB_5_bert+our_bilstm/results_run0_83
```
Notable arguments for testing are: 
* ```--mode```: test
* ```--model```: path to the model you want to test
* ```--output```: path to the folder you want to store the output that contains:
    * attention_output: word-level attention scores  
    * pred_seqs: predicted sequences
    * scores: detailed scores for each input file.

