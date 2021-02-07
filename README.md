# Forecasting Interaction Order on Temporal Graphs
 
## Abstract
Link prediction is a fundamental task of graph analysis and the topic has been studied extensively in the case of static graphs. Recent research interests in predicting links/interactions on temporal graphs. 
Most prior works employ snapshots to model graph dynamics and formulate the binary classification problem to predict the existence of a future edge. 
However, the binary formulation ignores the order of interactions and thus fails to capture the fine-grained temporal information in the data. 
In this paper, we propose a new problem on temporal graphs to predict the interaction order for a given node set (IOD).
We develop a Temporal ATtention network (TAT) for the IOD problem.
TAT utilizes fine-grained time information by encoding continuous time as fixed-length feature vectors.
For each transformation layer of TAT, we adopt attention mechanism to compute adaptive aggregations based on former layer's node representations and encoded time vectors.
We also devise a novel training scheme for TAT to address the permutation-sensitive property of IOD.
Experiments on several real-world temporal networks reveal that TAT outperforms the state-of-the-art graph neural networks by 55\% on average under the AUC metric.

# Usage
```
pip install -e . # first install as a local package
python -m TAT.main --save_log --desc 'desc' --dataset CollegeMsg --gpu 0 --model TAT
```

# Requirements

* python >= 3.7

* Dependency

```{bash}
scipy==1.5.0
torch==1.6.0
ipdb==0.13.4
numpy==1.18.5
scikit_learn==0.23.2
torch==1.6.0
torch_geometric==1.6.1
torch-scatter==2.0.5
torch-sparse==0.6.8
torch-spline-conv==1.2.0
torchvision==0.7.0
tqdm==4.47.0
pandas==1.05
pathlib2==2.3.5
```

# Helps
```
usage: Temporal GNNs. [-h] [--root_dir ROOT_DIR] [--checkpoint_dir CHECKPOINT_DIR] [--datadir DATADIR]
                      [--dataset {CollegeMsg,emailEuCoreTemporal,SMS-A,facebook-wall}] [--force_cache] [--directed DIRECTED]
                      [--gpu GPU] [--set_indice_length SET_INDICE_LENGTH] [--seed SEED] [--data_usage DATA_USAGE]
                      [--test_ratio TEST_RATIO] [--parallel] [--model {TAT,DE-GNN,GIN,GCN,GraphSAGE,GAT,TAGCN}] [--layers LAYERS]
                      [--hidden_features HIDDEN_FEATURES] [--dropout DROPOUT] [--negative_slope NEGATIVE_SLOPE]
                      [--use_attention USE_ATTENTION] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--lr LR] [--l2 L2]
                      [--optimizer OPTIMIZER] [--metric METRIC] [--perm_loss PERM_LOSS] [--in_features IN_FEATURES]
                      [--out_features OUT_FEATURES] [--prop_depth PROP_DEPTH] [--max_sp MAX_SP] [--sp_feature_type {sp,rw}]
                      [--time_encoder_type {tat,harmonic,empty}] [--time_encoder_maxt TIME_ENCODER_MAXT]
                      [--time_encoder_rows TIME_ENCODER_ROWS] [--time_encoder_dimension TIME_ENCODER_DIMENSION]
                      [--time_encoder_discrete {uniform,log}] [--time_encoder_deltas TIME_ENCODER_DELTAS] [--log_dir LOG_DIR]
                      [--save_log] [--debug] [--desc DESC] [--time_str TIME_STR]

optional arguments:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   Root directory
  --checkpoint_dir CHECKPOINT_DIR
                        Root directory
  --datadir DATADIR     Dataset edge file name
  --dataset {CollegeMsg,emailEuCoreTemporal,SMS-A,facebook-wall}
                        Dataset edge file name
  --force_cache         use cahced dataset if exists
  --directed DIRECTED   (Currently unavailable) whether to treat the graph as directed
  --gpu GPU             -1: cpu, others: gpu index
  --set_indice_length SET_INDICE_LENGTH
                        number of nodes in set_indice, for TAT Model
  --seed SEED           seed to initialize all the random modules
  --data_usage DATA_USAGE
                        ratio of used data for all data samples
  --test_ratio TEST_RATIO
                        test ratio in the used data samples
  --parallel            parallelly generate subgraphs
  --model {TAT,DE-GNN,GIN,GCN,GraphSAGE,GAT,TAGCN}
                        model name
  --layers LAYERS       largest number of layers
  --hidden_features HIDDEN_FEATURES
                        hidden dimension
  --dropout DROPOUT     dropout rate
  --negative_slope NEGATIVE_SLOPE
                        for leakey relu function
  --use_attention USE_ATTENTION
                        use attention or not in TAT model
  --epoch EPOCH         training epochs
  --batch_size BATCH_SIZE
                        mini batch size
  --lr LR               learning rate
  --l2 L2               l2 regularization
  --optimizer OPTIMIZER
                        optimizer (string)
  --metric METRIC       evaluation metric
  --perm_loss PERM_LOSS
                        weight of permutation loss
  --in_features IN_FEATURES
                        initial input features of nodes
  --out_features OUT_FEATURES
                        number of target classes
  --prop_depth PROP_DEPTH
                        number of hops for one layer
  --max_sp MAX_SP       maximum distance to be encoded for shortest path feature (not used now)
  --sp_feature_type {sp,rw}
                        spatial features type, shortest path, or random landing probabilities
  --time_encoder_type {tat,harmonic,empty}
                        time encoder type
  --time_encoder_maxt TIME_ENCODER_MAXT
                        time encoder maxt
  --time_encoder_rows TIME_ENCODER_ROWS
                        time encoder rows
  --time_encoder_dimension TIME_ENCODER_DIMENSION
                        time encoding dimension
  --time_encoder_discrete {uniform,log}
                        discrete type
  --time_encoder_deltas TIME_ENCODER_DELTAS
                        scale of mean time interval for discretization
  --log_dir LOG_DIR     log directory
  --save_log            save console log into log file
  --debug               debug mode
  --desc DESC           a string description for an experiment
  --time_str TIME_STR   execution time
```

