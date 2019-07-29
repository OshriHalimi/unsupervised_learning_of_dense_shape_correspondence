# "Unsupervised Learning Of Dense Shape Correspondence" (CVPR 2019)

**Authors: Oshri Halimi, Or Litany, Emanuele Rodola, Alex Bronstein, Ron Kimmel**

![](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/Picture1.png)

Link to the paper: http://openaccess.thecvf.com/content_CVPR_2019/html/Halimi_Unsupervised_Learning_of_Dense_Shape_Correspondence_CVPR_2019_paper.html

**If you use this code, please cite the paper.**
```
@inproceedings{halimi2019unsupervised,
  title={Unsupervised Learning of Dense Shape Correspondence},
  author={Halimi, Oshri and Litany, Or and Rodola, Emanuele and Bronstein, Alex M and Kimmel, Ron},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4370--4379},
  year={2019}
}
```
https://github.com/orlitany/DeepFunctionalMaps
## Getting Started
This tutorial will guide you how to run the experiments that appear in the paper **"Unsupervised Learning Of Dense Shape Correspondence" (CVPR 2019)**. The reader might be also interested in the supervised algorithm **"Deep Functional Maps:
Structured Prediction for Dense Shape Correspondence" (ICCV 2017)**, for which the code is provided in https://github.com/orlitany/DeepFunctionalMaps

### Code organization
Each folder contains the code and data for a specific experiment.
The repository is still updating, and I intend to finish organizing and uploading the whole code very soon :)
### Contact Information
For any questions regarding the code/paper, please contact me: **oshri.halimi@gmail.com**

## Single Pair Experiment (Self-supervised learning regime)
![](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/Capture.PNG)
In this experiment we show how the network can be optimized on a single pair of shapes and finally predict the correspondence.
The code for this experiment is located in the folder **\Single Pair Experiment**. 
### Data
The 3D models we use in this experiment are provided in the folder **\artist_models**.
Note that the ground-truth crrespondences are unknown for this pair.

### Preprocessing
First create an empty folder with the name **\tf_artist** in the base directory **\Single Pair Experiment**.
Then run the script preprocess_artist_model.mat.
This will create the input data to the network in the folder **\tf_artist**, namely the geodesic distance matrices, the LBO eigenfunctions and the initial SHOT descriptors will be calculated.

### Training the netowrk
Run **train_FMnet_self_supervised.py**.
When the train is finished the checkpoint file will be saved in **\Results\artist_checkpoint**.

### Predicting the correspondence
First create an empty folder with the name **\unsupervised_artist_results** under the **\Results** directory.
Then, run **test_FMnet_self_supervised.py**. The network predictions will be saved in **\Results\unsupervised_artist_results**.

### Visualization
To visualize the results run **visualize_results.m**.
The folder contains additional script files: **visualize_unsupervised_network_results.m** and **visualize_supervised_network_results.m** that visualize the correspondence predicted (1) by the unsupervised network that was trained with the (unannotated) single pair, and (2) by the supervised network that was trained on faust synthetic human shapes. We provide the final predictions with each of these methods in the folders **Unsupervised Network Results** and **Supervised Network Results**. Additionally we provide code for comparison with few axiomatic algorithms. To calculate the predictions with SGMDS or FM+SHOT algorithm, run **SGMDS.m** or **SHOT_FM.m**, respectively. Further details for the single-pair experiment are provided in the paper.

## Learning Correspondence of Synthetic Shapes (Unsupervised)

![](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/synFaustMen.jpg)

In this experiment we demostrate unsupervised learning of dense shape correspondence on a dataset of synthetic human shapes (MPI-FAUST). Link to the dataset: http://faust.is.tue.mpg.de/.
The code for this experiment is located in the folder **\Learning Correspondence of Synthetic Shapes\\** . 

### Data
The 3D models we use in this experiment are provided in the folder **\faust_synthetic\shapes**.
We provide the original .ply files, as well as our generated .mat files. This code processes 3D models with the specific format of the .mat files, containing the vertices coordinates in the field **VERT**, the faces triangulation in the field **TRIV**, and the number of vertices and faces in the fields **n** and **m**, respectively. To convert the 3D model .ply files to this specific .mat format, please run the script **convert_ply_to_mat.m** in the folder that contains the .ply files (in this case we already ran it in the folder **\faust_synthetic\shapes** and provided the generated .mat files). 

### Preprocessing
1. First, for each shape in faust synthetic shape collection we have to calculate the geodesic distance matrix.
To achieve this, please run the script **\faust_calc_distance_matrix.m**. This script will calculate the geodesic distance matrices and store them in a new folder **\faust_synthetic\distance_matrix**. This processing is done in parallel, using multiple matlab workers.
To view the geodesic distance from a source point, use the script **view_distance_matrix**. The result should be similar to this:
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/distance_map.PNG" width="300" />
</p>



2. Next, for each shape we'll calculate LBO eigenfunctions and the initial SHOT descriptors. Please run the script **\faust_preprocess_shapes.m**. The results will be saved in a new folder **\faust_synthetic\network_data**.

### Training the network
To train the network, please run **train_FMnet_unsupervised.py**. The trained network will be saved in the folder **\Results\train_faust_synthetic**. You can edit the number of training iterations, defined in **train_FMnet_unsupervised.py**, here we use 3K mini-batch iterations. 
* **Training loss**: During the training, we optimize the unsupervised loss and monitor the supervised loss for analysis purpose. When the training is finished, both losses are saves in **\Results\train_faust_synthetic\training_error.mat**. To visualize the losses during the training process - run the matlab script **visualize_synthetic_faust_test_results.m**. This script will produce a figure similar to figure 3 in the paper, showing a correlated decay of the unsupervised and the supervised losses. 
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/loss_graph_git.PNG" width="1000" />
</p>

### Predicting the correspondence on the test pairs
First create a list of all test pairs by calling the matlab script **create_test_pairs.m**. To test the network, please run **test_FMnet_unsupervised.py**, the test results will be saved in the folder **Results\test_faust_synthetic**.

### Visualization
To visualize the correspondence of a specific test pair, run the matlab script **visualize_synthetic_faust_test_results.m**.
Note that you should edit the indices of the test pair you wish to visualize in rows: 7,8,10. The result should look similar to this:
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/test_pair_vis.PNG" width="1000" />
</p>

### Geodesic error evaluation
* **One pair**:
To evaluate the error curve for a specific pair of shapes, you need the the predicted correspondence between the shapes, the ground truth correspondence, and the geodesic distance matrix of the target shape. To calculate the error curve use the command (matlab):
```
    errs = calc_geo_err(matches, gt_matches, D_model);
    curve = calc_err_curve(errs, 0:0.001:1.0)/100;
    plot(0:0.001:1.0, curve); set(gca, 'xlim', [0 0.1]); set(gca, 'ylim', [0 1])
```
The functions **calc_geo_err** and **calc_err_curve** are located in the folder **\Tools**. 

Here, **matches** is a vector such that 
```matches[i]=j```

where **i** is a source vertex and **j** is the corresponding vertex in the target shape, according to the algorithm result. 

Next, **gt_matches** is a vector such that ```gt_matches[i]=j```

where **i** is a source vertex and **j** is the corresponding vertex in the target shape, according to the ground truth.
For example, in synthetic faust: 
```gt_matches = 1:6890;```

Finally, **D_model** is the distance matrix of the target shape.

To plot the geodesic error curve for a test pair, run the script **calculate_geodesic_error_synthetic_faust_test_results.m** after editing the indices of the test pair in rows: 7,8,10,14. The result should be similar to this:
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/test_pair_geo.PNG" width="400" />
</p>

* **Shape collection**: To evaluate the error curve for a collection of shapes you should average the error curves of all pairs of shapes, as follows:
```
CURVES = zeros(N_pairs,1001);
for i=1:NumberOfPairs
   %here you calculate matches, gt_matches and D_model, for each pair
   ...
   errs = calc_geo_err(matches, gt_matches, D_model);
   curve = calc_err_curve(errs, 0:0.001:1.0)/100;
   CURVES(i,:) = curve;
end

avg_curve_unsupervised = sum(CURVES,1)/ NumberOfPairs;
plot(0:0.001:1.0, avg_curve_unsupervised,'r'); set(gca, 'xlim', [0 0.1]); set(gca, 'ylim', [0 1])
```

To plot the average geodesic error **over all the intra pairs** in faust synthetic dataset, run **calculate_geodesic_error_intra.m**. The result should look like this:
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/intra_geo.PNG" width="400" />
</p>

## Learning Correspondence of Real Scans (Unsupervised)
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/scans.png" width="500" />
</p>

TODO

## Learning Correspondence of PArtial Shapes (Unsupervised)
<p align="center">
  <img src="https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence/blob/master/partial.png" width="1000" />
</p>

TODO
### To be continued ...
