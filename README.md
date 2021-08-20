# MEUNet：Multi-scale Edge-based U-shape Network for Salient Object Detection

Code for paper "Multi-scale Edge-based U-shape Network for Salient Object Detection", by Han Sun, Yetong Bian , Ningzhong Liu, and Huiyu Zhou.

##### Requirements

- Python3.6
- Pytorch1.5
- torchvision
- numpy
- apex
- cv2

##### Usage

- Clone this repo into your workstation

```
git clone https://github.com/bellatong/MEUNet.git
```

##### training

1. Download the pre-trained model [resnet50](https://pan.baidu.com/s/1uuzzcs2bu6dzIJxfbTNsmw) ( password: 9yp3 )

2. Use  `edge.m`  to generate edge maps for the training set

3. Modify `train.py` to change both the dataset path and the file save path to your own real path

4. run `train.py`

	```
	python3 train.py
	```

	

##### test

1. Download our trained model [MEUNet](https://pan.baidu.com/s/1QyPmNkFoUf89i36v5rqSxg ) (password: 3zoZ) and put it into folder `out`

2. Modify the dataset path and file save path in the `test.py` and `metric/main_function.m` to your own real paths

3. run `test.py`, then the saliency maps will be generated under the corresponding path, and the evaluation scores for the model on the test dataset will be stored in `result.txt`

	```
	python3 test.py
	```

	

##### The result saliency maps

Here are saliency maps of our model on six different datasets (DUTS, ECSSD, DUT-OMRON, HKU-IS, PASCAL-S) [The result saliency maps](https://pan.baidu.com/s/1oqT5EI0vgCWTr1pXUl8PBw) (passwd: 6e21)
