# Deep Graph-based Spatial Consistency for Robust Non-rigid Point Cloud Registration

PyTorch implementation of the paper:

Deep Graph-based Spatial Consistency for Robust Non-rigid Point Cloud Registration.

[Zheng Qin](https://scholar.google.com/citations?user=DnHBAN0AAAAJ), [Hao Yu](https://scholar.google.com/citations?user=g7JfRn4AAAAJ), Changjian Wang, Yuxing Peng, and [Kai Xu](https://scholar.google.com/citations?user=GuVkg-8AAAAJ).

## Introduction

We study the problem of outlier correspondence pruning for non-rigid point cloud registration. In rigid registration, spatial consistency has been a commonly used criterion to discriminate outliers from inliers. It measures the compatibility of two correspondences by the discrepancy between the respective distances in two point clouds. However, spatial consistency no longer holds in non-rigid cases and outlier rejection for non-rigid registration has not been well studied. In this work, we propose Graph-based Spatial Consistency Network (GraphSCNet) to filter outliers for non-rigid registration. Our method is based on the fact that non-rigid deformations are usually locally rigid, or local shape preserving. We first design a local spatial consistency measure over the deformation graph of the point cloud, which evaluates the spatial compatibility only between the correspondences in the vicinity of a graph node. An attention-based non-rigid correspondence embedding module is then devised to learn a robust representation of non-rigid correspondences from local spatial consistency. Despite its simplicity, GraphSCNet effectively improves the quality of the putative correspondences and attains state-of-the-art performance on three challenging benchmarks.

## News

2023.02.28: This work is accepted by CVPR 2023. Code and models will be released soon.
