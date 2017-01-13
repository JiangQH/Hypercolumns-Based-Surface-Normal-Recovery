# Normal Recovery Project
--------
## Introduction
---------
This is a project to recovery surface normal from a monocular image. The main Idea of this project comes from this paper[*Marr Revisited: 2D-3D Alignment via Surface Normal Prediction*](http://www.cs.cmu.edu/~aayushb/marrRevisited) . You can view this project as an implimentation of that paper with some modifications, meanwhile I also extend the idea to outdoor scenes with success. 

Please Note that this project is done before the author release their code, so there are tiny differences with the net structure and a total different implementation of code in caffe.  I also provide the GPU implementation code of self-defined layer in caffe, so it brings a much faster training and deploying.

If you want to use the code or model to do your work or research, feel free to do it, it's opening, but do please site the original author.

## Result
-----
Here I show the result of both indoor and outdoor.     
The left is the rgb image, mid is the gt of normal and right is the model's prediction.   
**Indoor**   
![indoor](./result/indoor/concat.jpg)  

**Outdoor**   
![outdoor](./result/outdoor/concat.jpg)


##How to
----------
To run the code or model in this project, you have to install the **caffe** under my repository, It contains some self self-implementation code of caffe layer to help complete the project. (though there are some extra code in it you may not need, just leave it alone. I use them for other jobs==). clone the caffe by
```
git clone https://github.com/JiangQH/caffe.git
```
