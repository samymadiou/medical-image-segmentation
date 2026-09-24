# Medical Segmentation of a Heart

## 1. Introduction

Medical imaging plays an essential role in healthcare. It enables healthcare professionals to better analyze and diagnose their patients. The study of medical images depends primarily on visual interpretation and the experience of radiologists. However, this can be a tedious task for physicians. For example, images may be of poor quality, and it may be difficult to locate a tumor.

The use of computer-assisted systems can therefore be very useful and interesting for avoiding these problems. Image segmentation is a critical image-analysis technique. Its purpose is to extract information from an image. In the medical field, image segments often correspond to different categories of tissue, organs, or other biological structures in the human body. Thus, systems based on deep machine-learning models can assist specialists by detecting potential tumor regions, with a certain error rate.

This report summarizes the work carried out for the MTI881 course at Ecole de Technologie Superieure in Montreal. The purpose of the exercise was to implement the best neural network for cardiac image segmentation. More precisely, the objective of this project was to segment the heart into three parts: the right ventricle (RV), the left ventricle (LV), and the myocardium (MYO).

## 2. Related Work

### The UNet family

The UNet model is one of the most widely used models for medical image segmentation because of its simplicity. Several variants of this model exist, including ResidualUNet, TransUNet [1], and SwinUNet [2]. The latter two models are among the best-performing models for medical image segmentation. More specifically, the TransUNet model proposed by Chen et al. [1] and the SwinUNet model proposed by Cao et al. [2] placed third and second, respectively, in the ACDC challenge [3][4].

## 3. Methodology

Many options were available when choosing a learning model for the challenge. However, it was first necessary to consider the nature of the input images before selecting a model. The images are 256 x 256 and are 2D. In addition, our label consists of four classes: [Background, LV, RV, MYO], where the background has different properties from the other classes. The training set contains 1,208 images, and the validation set contains 90 images.

### 3.1 Data Augmentation

Data augmentation is a very popular technique in machine learning. It is used to increase the quantity and quality of the data available for training a model. In practice, it consists of generating new data from existing data by using different transformation and modification techniques.

One of the main advantages of data augmentation is that it helps avoid overfitting. Data augmentation reduces the risk of overfitting by providing the model with a greater variety of training data. By augmenting the data, we can also train machine-learning models more effectively and obtain better results.

We decided to use this technique because we had little data. We had only 1,208 training images, which is clearly insufficient. For this purpose, we used the Python library Albumentations [2], which provides several techniques and transformations for data augmentation.

**RandomRotate90:** This transformation randomly rotates an image by 0, 90, 180, or 270 degrees.

*Figure 1: Random rotation*

**VerticalFlip and HorizontalFlip:** These two functions flip the image about the vertical and horizontal axes, producing a mirror effect.

*Figure 2: Vertical and horizontal flip*

**CenterCrop:** The CenterCrop function crops an image while retaining only its central portion. This means that the edges of the image are removed and only the central part is kept. We chose to remove 32 pixels from the image's length and width.

*Figure 3: Center crop*

**RandomBrightness:** This transformation randomly adjusts the brightness of an image, either increasing or decreasing it.

*Figure 4: Random brightness*

**RandomContrast:** This transformation randomly adjusts the contrast of an image, either increasing or decreasing it.

*Figure 5: Random contrast*

**MotionBlur:** This transformation adds motion blur to an image.

*Figure 6: Motion blur*

**GaussNoise:** This transformation adds Gaussian noise to an image. Gaussian noise is a type of random noise that follows a normal distribution.

*Figure 7: Gaussian noise*

**CoarseDropout:** This function masks different random regions in the image.

*Figure 8: Coarse dropout*

These transformations were randomly applied to the images and their labels with a probability of 0.4. Thus, there was a 40% chance that each transformation would be applied to an image. Through data augmentation, we were able to train the model with nearly 5,000 images.

### 3.2 Model Architectures

The following are the architectures that we developed or considered developing.

#### 3.2.1 UNet

This model has slightly more than 30 million parameters, depending on our configuration. It was developed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. The model consists of an encoder and a decoder. The encoder block contains a series of convolutional layers that reduce the dimensions of the input image. The decoder block contains upsampling layers and a series of convolutional layers [5].

*Figure 9: UNet model architecture*

#### 3.2.2 TransUNet

The TransUNet architecture is very similar to UNet. This model also contains encoder and decoder blocks, as well as skip connections between the encoder and decoder layers. Unlike UNet, an image-to-image translation is performed before predicting the segmentation mask. The image is translated into a synthetic domain in which objects are more clearly visible and easier to segment.

*Figure 10: TransUNet model architecture*

#### 3.2.3 ResidualUNet

Once again, ResidualUNet is not very different from UNet. The difference between the two models is that ResidualUNet contains residual layers, which allow the model to learn the residual between the input and the output of a layer.

These residual layers also help avoid the vanishing-gradient problem. This phenomenon can occur during training when the gradients of the parameters with respect to the loss function tend toward zero. It can happen when parameters are initialized with large values or when the network architecture contains a large number of layers. This is the case with ResidualUNet, which contains slightly more than 70 million parameters in our configuration.

In short, because of the limited time available to develop a functional model that was sufficiently performant to compete in the ACDC challenge, we only developed and tested UNet and ResidualUNet.

*Figure 11: ResidualUNet model architecture*

#### 3.2.4 SwinUNet

The difference between SwinUNet and UNet is that SwinUNet performs image classification using a self-supervised-learning approach before predicting the segmentation mask.

*Figure 12: SwinUNet model architecture*

### 3.3 Hyperparameters

Hyperparameters are parameters that control the learning process. They are selected by developers before model training begins. It is crucial to choose the correct values in order to obtain the best model performance on images that the model has never seen.

In our case, we used hyperparameters at several levels to configure our model. First, we used the batch size, which determines the number of image samples used per iteration during training and validation. The model is trained on one batch of data at a time.

After several attempts, we noticed that the larger the batch size, the more quickly the model could process large groups of images. However, this could cause instability during convergence and increase the model's error. We therefore used a batch size that was large enough to allow rapid convergence but small enough to avoid increasing the error and to improve performance. After testing different batch sizes for training and validation, we finally chose batches of 8-12 images for training and 3-6 images for validation.

We also had to find the right settings for our loss function. Our loss function is a combination of two loss functions, each with a weight $w$. This weight allows the model to receive a greater penalty when it makes incorrect predictions. In our situation, we wanted to increase the penalty associated with Dice Loss so that the model would focus on similarity to the segmentation mask. After testing several weight settings, we chose $w_{DiceLoss}=0.6$ and $w_{CE}=0.4$. These settings considerably changed the performance of our UNet and ResidualUNet models. The models achieved approximately 83-85% mean Dice on the test images.

Next, we had to choose a learning rate. The learning rate is the most important setting in machine learning because it controls how quickly the model learns from the training images. Choosing it is crucial because it can greatly affect learning. A rate that is too high can prevent the model from converging while minimizing the losses. Conversely, a rate that is too low can make convergence take too long. A very small rate can also cause the model to remain stuck at a local minimum.

Consequently, the learning rate must be high enough to converge quickly but low enough to train the model properly. There is no universal solution for finding the correct learning rate because it depends on several factors, such as model complexity and the quality of the training images. The best strategy for finding the optimal rate is trial and error until the rate that provides the best performance is found. After using several learning-rate optimization techniques, we reached $lr=0.001$ with the Adam optimizer.

In summary, hyperparameters are very important parameters for model training. They have a major impact on model performance. To find the best combination of parameters, several tests must be performed with different settings in order to obtain a model that performs well on the test set. This was the longest phase of the project, and in our opinion, different settings might have produced better performance.

### 3.4 Loss Functions

To evaluate our learning model, we need a loss function that measures the model's error by comparing its predictions with the ground-truth (GT) images. This function adjusts the model's weights to minimize the error. The loss function is used together with gradient descent to propagate the error through the model and adjust its parameters in order to minimize the loss.

At first, we used only cross-entropy as the loss function during training. However, this was not optimal because it did not maximize the Dice similarity metric. We therefore added Dice Loss to our loss function and combined it with cross-entropy to maximize the similarity score [6].

$$
Loss = w_{dice} * L_{dice} + w_{CE} * L_{CE}
$$

$$
L_{dice}: 1 - Dice(s_{pred}, onehotlabels)
$$

$$
L_{CE}: CrossEntropy(pred, labels)
$$

### 3.5 Optimizers

Optimizers [7] are essential in machine learning because they improve convergence toward global minima. There are many optimizers, each with its own advantages and disadvantages. We decided to use the Adam and SGD optimizers combined with Momentum. In addition, to vary the learning rate and improve convergence, we used different types of schedulers.

The SGD optimizer is an algorithm used to improve the convergence and performance of our model. It provides better model generalization, but it can take a long time to converge. Using schedulers is effective for increasing the speed of convergence.

We tested the optimizer with a custom scheduler [7] that changes the learning rate by a factor at each epoch.

*Figure 13: Training with the SGD optimizer*

*Figure 14: Lambda LR*

The disadvantage of this custom scheduler is that after a certain number of epochs, the learning rate becomes very small and the model stops learning.

We also tried ReduceLROnPlateau [7]. This scheduler decreases the learning rate when the validation error begins to plateau after a certain number of epochs. It is very effective because it automatically decreases the rate when the error stagnates at a given level. The most important aspect of this scheduler is choosing the correct patience value. If the patience value is too large, the model may overfit before the learning rate is reduced.

*Figure 15: Reduce-on-plateau LR*

We also tested a cyclic scheduler [7]. It defines a cyclic learning rate that varies in a triangular pattern during model training. This can lead to faster convergence by taking the shortest path to the global minimum. With a standard scheduler, the model would take a long detour before reaching the global minimum. The graph below illustrates this behavior.

*Figure 16: Cyclic LR*

In the end, we decided to keep the Adam optimizer [7] because it does not really need a scheduler, since the scheduler only manages the learning rate. Adam is widely used by the scientific community. Although it converges quickly, it does not provide better generalization than SGD. However, it provides very satisfactory performance. The Adam optimizer without a scheduler produced the best predictions.

*Figure 17: Adam*

## 4. Results

During training, we tested only the UNet and ResidualUNet models, using different configurations. The following table shows the Dice scores obtained for each model and each class.

| Model | Parameters | Class 1 | Class 2 | Class 3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| ResidualUNet | lr=0.01, opt=Adam, epoch=20 | 0.83 | 0.84 | 0.93 | 0.8700 |
| ResidualUNet | lr=0.01, opt=SGD, epoch=33 | 0.79 | 0.74 | 0.88 | 0.8033 |
| UNet | lr=0.001, opt=SGD+CyclicLr, epoch=22 | 0.80 | 0.79 | 0.90 | 0.7233 |
| UNet | lr=0.001, opt=SGD+LambdaLr, epoch=22 | 0.69 | 0.68 | 0.80 | 0.7233 |
| UNet | lr=0.005, opt=Adam, epoch=62 | 0.83 | 0.82 | 0.92 | 0.8567 |
| UNet | lr=0.001, opt=Adam, epoch=30 | 0.85 | 0.85 | 0.93 | 0.8767 |

*Table 1: Dice results obtained on the test images*

As can be seen, both models performed very well. We spent considerably more time training ResidualUNet because it had nearly twice as many parameters. For the challenge, we ultimately chose the UNet model because the values returned by its loss function were lower.

| Metric | Class 1 | Class 2 | Class 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Dice | 0.67 | 0.77 | 0.85 | 0.76 |
| HD | 8.41 | 5.64 | 4.57 | 6.21 |
| ASD | 2.51 | 1.93 | 1.79 | 2.07 |

*Table 2: Results obtained during the challenge*

## 5. Conclusion

In conclusion, we chose to use the UNet model. We applied data augmentation to this model, which increased the amount of data available for training. We had to use different techniques, such as RandomRotate90, CenterCrop, and GaussNoise, in order to obtain quality data.

In addition, thanks to the hyperparameters, the UNet model achieved the best results during training, with a mean Dice score of 0.87. However, it would be interesting to evaluate our model's performance on other tasks in the same medical-imaging domain. We plan to propose a more advanced implementation in order to maximize our chances of achieving better results.

## References

[1] Olivier Bernard, Alain Lalande, Clement Zotti, Frederick Cervenansky, Xin Yang, Pheng-Ann Heng, Irem Cetin, Karim Lekadir, Oscar Camara, Miguel Angel Gonzalez Ballester, et al. Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved? *IEEE Transactions on Medical Imaging*, 37(11):2514-2525, 2018.

[2] Alexander Buslaev, Vladimir I. Iglovikov, Eugene Khvedchenya, Alex Parinov, Mikhail Druzhinin, and Alexandr A. Kalinin. Albumentations: Fast and flexible image augmentations. *Information*, 11(2), 2020.

[3] Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, and Manning Wang. Swin-Unet: Unet-like pure transformer for medical image segmentation. arXiv preprint arXiv:2105.05537, 2021.

[4] Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, and Yuyin Zhou. TransUNet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.

[5] Huimin Huang, Lanfen Lin, Ruofeng Tong, Hongjie Hu, Qiaowei Zhang, Yutaro Iwamoto, Xianhua Han, Yen-Wei Chen, and Jian Wu. UNet 3+: A full-scale connected UNet for medical image segmentation. In *ICASSP 2020 - 2020 IEEE Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 1055-1059. IEEE, 2020.

[6] Shruti Jadon. A survey of loss functions for semantic segmentation. In *2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)*, pages 1-7. IEEE, 2020.

[7] Aliasghar Mortazi. Optimization algorithms for deep learning based medical image segmentations. 2019.