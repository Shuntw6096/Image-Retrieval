# Image Retrieval using Fashion MNIST dataset
NTUT-EE Deep Learning 2020 project-1

Project Introduction
------
Image Retrieval（圖像搜索）是autoencoder（自編碼器）的應用之一，嚴格來說，是基於內容的圖像搜索（Content-based image retrieval，CBIR），透過圖像彼此的相似度來搜尋圖像，我們利用encoder將圖像表達成一組向量的方式計算圖像的相似程度，我們利用歐氏距離（Euclidean Distance）表達圖像的相似程度。我們利用DNN組成autoencoder並利用Adam優化器訓練，訓練過程使用Large Mini-Batch訓練（Batch size=256），並動態遞減Learning Rate。我們嘗試使用ZCA Whitening（Zero-phase Component Analysis Whitening）做圖像預處理想要縮減網路規模相比於沒有將訓練集預處理訓練的網路。

Dataset Introduction
------
Fashion MNIST已經預先劃分資料用途，訓練集60000筆、測試集10000筆，總共10個類別：T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot對應0~9類別．每個類別6000筆，測試集則每個類別1000筆．


Data Processing
------
**ZCA Whitening（Zero-phase Component Analysis Whitening）**  
首先計算訓練集的協方差矩陣![formula0](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula0.JPG)  
並對此做奇異值分解（singular value decomposition）![formula0](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula1.JPG)  
U是Σ的特徵向量矩陣，S是其特徵值矩陣；因为Σ 是對稱方陣![formula5](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula5.JPG)  
![formula2](https://github.com/Shuntw6096/Image-Retrieval/blob/use_tensorboard_0421/img/formula2.JPG)  


