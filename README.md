#   template of DNN in tensorflow

## requirement

```
1. python3
2. python3-tensorflow
3. python3-tensorboard
4. python3-numpy
```
##  run

### prepare

1.  add/delete args if needed in **./train.py** function **get\_args**
2.  change loss function, result, input layer & ground truth placeholder in **./network.py** class **Network**, (function **net**, **func\_loss**, **func\_result**)
3.  add new network architecture in **./model/[name].py** and import it in **./model/\_\_init\_\_.py**
4.  use new network architecture in **./network.py** class **Network** function **hidden\_layer**
5.  change input data in **./data\_provider.py** function **get\_\*\*\*\***, the keys of dict should be corresponding to the names of placerholder in **./network.py** class **Network** function **net**

### train
```shell
$   python3 -h # to see args
```
```
Training of CNN

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the task
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  -s STEP, --step STEP  total training step, -1 means infinite
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size of each step
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        path to load pretrained model, default for no loading
  --val-step VAL_STEP   how many training steps before each validation
  --save-step SAVE_STEP
                        how many training steps before each model-saving
  --model-path MODEL_PATH
                        path to save model
  --logdir LOGDIR       path to save training logs

```

```shel
$   python3 train.py [params] # to run
```

### see run log

```shell
$   tensorboard --logdir=train_log
```


## file logic

* ./train.py
    - get args
    - load network class from **./network.py**
* ./network.py
    - get input data and ground truth from function **get\_train** & **get\_val** & **get\_test** from **./data\_provider.py**
    - load hidden layer from **./model/\_\_init\_\_.py**
* ./model/\_\_init\_\_.py
    - load models from **./model/\*.py**
 

## file tree
* cnn  
    + train.py # training scripts
    + network.py # network class
        - class Network
            -  func_result # function, output the result of net
            -  func_loss # loss function
            -  hidden_layer # function, hidden_layer of the net, import from ./model/
            -  net # network, the network
            -  train # train function
        - cfg # config of input & output data, defined in ./commom.py
        - get\_train # function, get train data from ./data\_provider.py
        - get\_val # the same
        - get\_test # the same
    + commom.py # config of image shape and something
        - data\_provider.py # provide data (not complete yet)
    + README.md # me
    + Makefile # some commonds
    + IO.py # read data from dir data/
    + model/ # store basic models or new model   
        - \_\_init\_\_.py # import model from dir model/, like line #1
        - demo.py # demo model
    + scripts/ # store some shells
    + data/ # store data
    + train\_log/ # training log (default path)
    + models/ # store training models
    + train\_events/ # store tensorboard logs



