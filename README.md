# "Unsupervised Learning Of Dense Shape Correspondence" (CVPR 2019)

**Authors: Oshri Halimi, Or Litany, Emanuele Rodola, Alex Bronstein, Ron Kimmel**

![](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/Picture1.png)

link to the paper: http://openaccess.thecvf.com/content_CVPR_2019/html/Halimi_Unsupervised_Learning_of_Dense_Shape_Correspondence_CVPR_2019_paper.html

**If you use this code, please cite the paper.**

@inproceedings{halimi2019unsupervised,
  title={Unsupervised Learning of Dense Shape Correspondence},
  author={Halimi, Oshri and Litany, Or and Rodola, Emanuele and Bronstein, Alex M and Kimmel, Ron},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4370--4379},
  year={2019}
}
## Getting Started

These instructions will guide you how to run the experiments that appear in the paper.
Each folder contains the code and data for a specific experiment.
The repository is still updating, for now I provide the code for the single-pair experiment, I intend to finish uploading the whole code very soon.

## Single Pair Experiment (Self-suspervised learning of dense shape correspondence)
In this experiment we show how the network can be optimized on a single pair of shapes and finally predict the correspondence.
The code for this experiment is located in the folder **\Single Pair Experiment**. 
### Data
The 3D models we use in this experiment are provided in the folder **\artist_models**.
Note that the ground-truth crrespondences are unknown for this pair.

### Preprocessing
First create an empty folder with the name **\tf_artist** in the base directory \Single Pair Experiment.
Then run the script preprocess_artist_model.mat.
This will create the input data to the network in the folder **\tf_artist**, namely the geodesic distance matrices, the LBO eigenfunctions and the initial SHOT descriptors will be calculated.

### Training the netowrk
Run **train_FMnet_self_supervised.py**.
When the train is finished the checkpoint file will be saved in **\Results\artist_checkpoint**.

### Predicting the correspondence
First create an empty folder with the name **\unsupervised_artist_results** under the **\\Results** directory.
Then, run **test_FMnet_self_supervised.py**. The network predictions will be saved in **\Results\unsupervised_artist_results**.

