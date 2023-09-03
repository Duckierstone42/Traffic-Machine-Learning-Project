# Traffic-Machine-Learning-Project
This repo contains a program for sorting German traffic signs using a CNN from the Keras API in Tensorflow. There are two folders of images, gtsrb, and gtsrb-small, one with 43 different traffic signs and the other with 3. The CNN structure consists of two Convolutional 2D layers each followed by Pooling laters, and then a couple of dense layers. Dropout regularization is used, with the adam optimizer. The dataset used can be downloaded [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

In order to run the program, run the following command in the command prompt:
```bash
python traffic.py gtsrb
```
If you want to test it out with the smaller dataset, simply replace gtsrb with gtsrb-small.

The following output is to be expected from the program:
```
Epoch 1/10
500/500 [==============================] - 7s 13ms/step - loss: 2.6234 - accuracy: 0.4949  
Epoch 2/10
500/500 [==============================] - 6s 13ms/step - loss: 0.6084 - accuracy: 0.8288
Epoch 3/10
500/500 [==============================] - 6s 12ms/step - loss: 0.3709 - accuracy: 0.8962
Epoch 4/10
500/500 [==============================] - 6s 12ms/step - loss: 0.2602 - accuracy: 0.9276
Epoch 5/10
500/500 [==============================] - 6s 12ms/step - loss: 0.2053 - accuracy: 0.9449
Epoch 6/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1924 - accuracy: 0.9475
Epoch 7/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1633 - accuracy: 0.9550
Epoch 8/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1421 - accuracy: 0.9612
Epoch 9/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1659 - accuracy: 0.9574
Epoch 10/10
500/500 [==============================] - 6s 12ms/step - loss: 0.0988 - accuracy: 0.9740
333/333 - 1s - loss: 0.2284 - accuracy: 0.9562 - 1s/epoch - 4ms/step
```
The program detects the correct traffic sign with about 96% accuracy from the 43 possible options, in the above case.
