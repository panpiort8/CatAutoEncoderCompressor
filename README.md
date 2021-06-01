# Cat Auto Encoder Compressor (CAEC)

An image compressor based on an auto-encoder architecture designed for compressing images of cats.

## Usage
Compression:
```
python -m src.scripts.compress --config configs/compressor.yaml \
-s datasets/test/cat_1.bmp -t cat_1.cat
```
Decompression:
```
python -m src.scripts.decompress --config configs/compressor.yaml \
-s cat_1.cat -t cat_1.bmp
```

## Methods

Obviously, the code is based on the parent repository (https://github.com/alexandru-dinu/cae).

### Architecture

The compressor is based on an auto-encoder architecture inspired by [1] and [2]. It is a lightweight convolutional neural network working on 128x128 patches of an image. It can therefore be easily adapted to work with images of arbitrary size.

### Quantization

I have tested two methods of quantization:
* Stochastic binarization [2]: 
  ![img_1.png](images/stochastic.png)
* Uniform quantization [3]:
  
    ![img_3.png](images/uniform.png)

### Dataset

The model was trained and tested on ~10k images of cats taken from [Kaggle cat dataset](https://www.kaggle.com/crawford/cat-dataset), resized to 1280x768 with [Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling). The dataset was randomly splitted into two subsets:`train` and `test` with ratio 95:5.

### Experimental setup

## Objective comparison

## Subjective comparison

## References

- [1] [Lossy Image Compression with Compressive Autoencoders, Theis et al.](https://arxiv.org/abs/1703.00395)
- [2] [Variable Rate Image Compression with Recurrent Neural Networks, Toderici et al.](http://arxiv.org/abs/1511.06085)
- [3] [An Autoencoder-based Learned Image Compressor:
Description of Challenge Proposal by NCTU, Alexandre et al.](https://arxiv.org/abs/1902.07385)
