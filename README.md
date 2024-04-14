# Stereo Vision for 3D Reconstruction

## Project Overview

This project implements stereo vision to achieve 3D reconstruction from two different camera angles. By comparing two images of the same scenario, we extract the relative positions of objects in 3D space. The project involves the calibration process to compute fundamental and essential matrices using correspondence points, image rectification, correspondence matching, and depth calculation.

## Table of Contents
- [Calibration](#calibration)
- [Rectification](#rectification)
- [Correspondence](#correspondence)
- [Depth Calculation](#depth-calculation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)


## Calibration

The calibration phase involves matching images using the SIFT algorithm and Brute-Force Matcher, extracting correspondences, and computing the fundamental matrix. This phase is critical for establishing a relationship between two camera perspectives and involves normalizing image points, estimating the camera pose, and finding the best point correspondences using RANSAC.

## Rectification

Rectification transforms the images to align them along their epipolar lines. This process involves image warping and homographic transformations to rectify the images and facilitate easier point correspondence.

## Correspondence

Correspondence involves searching along epipolar lines in the second image for matches to points in the first image. Block matching and the Sum of Absolute Differences (SAD) approach are used to compute disparity for the rectified image pair.

## Depth Calculation

Depth calculation is the final step where 3D information is extracted using camera parameters and disparity data. The depth for each set of images is fine-tuned for accuracy.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Usage

To run this project, clone the repository and install the dependencies listed above. The main scripts are `artroom.py`, `chess.py`, and `ladder.py`, corresponding to different datasets.

```bash
python artroom.py
python chess.py
python ladder.py
```

## Results

## Calibration
Here's an example image of feature matching for Artroom Dataset:

![Feature Matching](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/chess_matched_image.png)

## Rectification
This image shows the results of image rectification:

![Rectified Image](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/rectified_epi_polar_lines_.png)

## Correspondence
Here are the epipolar lines demonstrating correspondence:

![Epipolar Lines](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/epi_polar_lines_.png)

## Depth Calculation
The following images illustrate the depth calculation results:

Depth Image - Gray:

![Depth Image Gray](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/disparity_image_gray.png)


Depth Image - Heat Map:

![Depth Image Heat](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/depth_image_heat.png)

## Disparity
The following images illustrate the Disparity results:

Disparity Image - Gray:

![Disparity Image Gray](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/disparity_image_gray.png)


Disparity Image - Heat Map:

![Disparity Image Heat](https://github.com/kalavagunta-vamshi/673-p4/blob/main/results/artroom/disparity_image_heat.png)


