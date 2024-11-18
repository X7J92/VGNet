# VGNet
========
This is the official implementation of [Phrase Decoupling Cross-Modal Hierarchical Matching and Progressive Position
Correction for Visual Grounding].The code is still being organized, please be patient.


## Training

The following is an example of model training on the referit dataset.
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --config configs/VGNet_R101_referit.py
```
We train the model on 4 GPUs(4090) with a total batch size of 64 for 90 epochs. 
The model and training hyper-parameters are defined in the configuration file ``VGNet_R101_referit.py``. 
We prepare the configuration files for different datasets in the ``configs/`` folder. 


## Evaluation
Run the following script to evaluate the trained model with 4 GPUs.

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env test.py --config configs/VGNet_R101_referit.py --checkpoint VGNet_R101_referit.pth --batch_size_test 16 --test_split val
```


## e-mail

call me: mengzhaowangg@163.com



## Acknowledgement
Part of our code is based on the previous works [DETR](https://github.com/facebookresearch/detr) and [TransVg](https://github.com/djiajunustc/TransVG).
