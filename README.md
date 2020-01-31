# Hierarchical Convolutional Attention Networks implementation in Tensorflow

This is an implementation of the neural network presented in the article ["Hierarchical Convolutional Attention Networks for Text Classification"](https://www.aclweb.org/anthology/W18-3002.pdf)

## Environement 
The project is developed using Python 3.7, Tensorflow 1.15.0 and Keras on Tensorflow 2.2.4-tf.

## How to build a HCAN
To build a Keras HCAN model, you must execute the `build_HCAN` function in `src/HCAN.py`

## Model Architecture
![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0933365719303562-gr1.jpg)

## How to run the demo
A small demo is given in the project. To run it you must : 
1. Go inside `demo` directory
2. Modify the variable `embedding_root_dir` inside `setup.py`. This variable represents the path to the directory where to download and unzip pretrained Glove words' embeddings.
3. Execute `setup.py` (`python3 setup.py` from command line). If you have already downloaded the words' embeddings, skip this step.
4. Run `Demo_HCAN.ipynb`

## License
Copyright 2020 Nabih Nebbache and Camille Fossier

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
