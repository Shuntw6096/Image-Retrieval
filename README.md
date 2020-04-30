# Image Retrieval using Fashion MNIST dataset
NTUT-EE Deep Learning 2020 project-1

# Project Introduction

Image Retrieval（圖像搜索）是autoencoder（自編碼器）的應用之一，嚴格來說，是基於內容的圖像搜索（Content-based image retrieval，CBIR），透過圖像彼此的相似度來搜尋圖像，我們利用encoder將圖像表達成一組向量的方式計算圖像的相似程度，我們利用歐氏距離（Euclidean Distance）表達圖像的相似程度。我們利用DNN組成autoencoder並利用Adam優化器訓練，訓練過程使用Large Mini-Batch訓練（Batch size=256），並動態遞減Learning Rate。我們嘗試使用ZCA Whitening（Zero-phase Component Analysis Whitening）做圖像預處理想要縮減網路規模相比於沒有將訓練集預處理訓練的網路。

# Dataset Introduction

Fashion MNIST已經預先劃分資料用途，訓練集60000筆、測試集10000筆，總共10個類別：T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot對應0~9類別．每個類別6000筆，測試集則每個類別1000筆．


# Data Processing

**ZCA Whitening（Zero-phase Component Analysis Whitening）**  
首先計算訓練集的協方差矩陣![formula0](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula0.JPG)  
並對此做奇異值分解（singular value decomposition）![formula0](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula1.JPG)  
U是Σ的特徵向量矩陣，S是其特徵值矩陣；因为Σ 是對稱方陣![formula4](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula4.JPG)  
![formula2](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula2.JPG)  
加入ϵ是為了避免特徵值接近零導致縮放時除以零，然後ZCA Whitening與PCA Whitening的關係是：![formula3](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula3.JPG)  
通常使用PCA Whitening是為了去除特徵間彼此的相關性，而ZCA Whitening會將PCA處理過的數據變換進原本的空間，通常會在計算協方差矩陣前減去各特徵的均值。  
準備四種訓練集：  
|#|圖片|說明|
|---|----|:---:|
|1|![original img](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/original_img.JPG)|原始圖片|
|2|![adding noise img](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/img_add_noise.JPG)|原始圖片加入高斯噪聲|
|3|![zca img](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/img_zca.JPG)|原始圖片經過ZCA Whitening|
|4|![zca noise img](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/img_add_noise_zca.JPG)|原始圖片先加入高斯噪聲再經過ZCA Whitening|

# Neural Networks Structure

|#|Autoencoder|encoder|decoder|
|---|----|---|---|
|Deep Networks|![autoencoder deep](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/autoencoder_deep.jpg)|![encoder deep](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/encoder_deep.jpg)|![decoder deep](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/decoder_deep.jpg)|
|Non-deep Networks|![autoencoder non deep](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/autoencoder_non_deep.jpg)|![encoder deep](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/encoder_non_deep.jpg)|![decoder deep](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/decoder_non_deep.jpg)|  

# Experiments
** * Deep NN **
**1. 使用原始圖片當訓練集： **
   
|#|圖片|說明|
|---|----|:---:|
|1|![deep nn 1](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/deepnn1.jpg)|圖片重建|
|2|![deep nn 2](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/deepnn2.jpg)|使用高斯噪聲圖片重建|
|3|![deep nn 3](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/deepnn3.jpg)|使用原始圖片做圖片搜尋|
|4|![deep nn 4](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/deepnn4.jpg)|使用噪聲圖片做圖片搜尋|
|5|![deep nn 5](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/deepnn5.jpg)|原始測試集搜尋原始訓練集的分類混淆矩陣|
|6|![deep nn 6](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/deepnn6.jpg)|原始測試集搜尋有噪聲的訓練集的分類混淆矩陣|

