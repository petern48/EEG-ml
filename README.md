# EEG-ml

Contributors:
Peter Nguyen  
UID: 205954558  
Email: petern0408@g.ucla.edu

Sean Tang  
UID: 905991152  
Email: seantang001@g.ucla.edu

Oliver Wang  
UID: 106021703  
Email: owang22@ucla.edu

Jingchao Luo  
UID: 205965873  
Email: jingchao.luo@ucla.edu

## Keras
- `keras/CNN.ipynb` Keras CNN implementation of EEGNet
- `keras/Hybrid-CNN-GRU.ipynb` Keras Hybrid CNN GRU implementation
- `keras/LSTM_GRU.ipynb](keras/LSTM_GRU.ipynb` Keras LSTM and GRU implementations
- `keras/ensemble.ipynb](keras/ensemble.ipynb` Keras Ensembling of the above three architectures using the pretrained models in `pretrained_models/`
- `keras/lstm_acc_vs_time.ipynb](keras/lstm_acc_vs_time.ipynb` Accuracy vs time experiment using keras LSTM model
- `keras/optimize_one_subject.ipynb](keras/optimize_one_subject.ipynb` Experiment optimizing model using training data from one subject at a time
- `keras/resnet.ipynb](keras/resnet.ipynb` Keras ResNet implementation
## Torch
- `torch/acc_vs_time/torch_acc_vs_time.ipynb` Keras CNN implementation of EEGNet
- `torch/best_impl` Torch CNN implementation fo EEGNet. See notebook: https://drive.google.com/file/d/1Bm5LhdbgtMoytSfJ_-QDqKck6l1LCu_F/view?usp=sharing
- `torch/filter_vis.ipynb` Visualization of Torch CNN implementation fo EEGNet: https://colab.research.google.com/drive/1WIHZ2SBC6YcoISVA5EIc1f2kcipgBW9i?usp=sharing

## Classic/None-CNN

- `classic_noncnn/tsmixer.ipynb` SOTA None CNN model for MTSC
- `classic_noncnn/classic_methods.ipynb` ROCKET model from sktime
