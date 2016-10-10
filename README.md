# Project description.  
This is a project to recovery the normal from monocular image. Mainly the implementation of this paper[*Marr Revisited: 2D-3D Alignment via Surface Normal Prediction*](http://www.cs.cmu.edu/~aayushb/marrRevisited) with some modifications.This project is done before the author release the code, with tiny difference in the network and different implementation of code in caffe.  
  
It achieves the same result and being much faster since the GPU version code of self-defined caffe layer has also been implemented.Using hypercolumn features sampled from a pre-trained VGG-16 Network and formulate the problem as a regression network, it achieves a state-of-art result, which can be seen in the eval folder.
