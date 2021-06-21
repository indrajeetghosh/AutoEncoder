# AutoEncoder

An autoencoder neural network is an unsupervised learning algorithm that applies backpropagation, setting the target values to be equal to the inputs. "Autoencoding" is a data compression algorithm where the compression and decompression functions are 1) data-specific, 2) lossy, and 3) learned automatically from examples rather than engineered by a human. Autoencoders are data-specific, which means that they will only be able to compress data similar to what they have been trained on. Autoencoders are lossy, which means that the decompressed outputs will be degraded compared to the original inputs.  Autoencoders are learned automatically from data examples, which is a useful property: it means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input. It doesn't require any new engineering, just appropriate training data.

In this code, I showed simple autoencoder with multiple data inputs (data parallel function) for HAR on pytorch framework. 


Reference: -
1). https://blog.keras.io/building-autoencoders-in-keras.html
2). https://www.deeplearningbook.org/contents/autoencoders.html
3). http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
4). https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
