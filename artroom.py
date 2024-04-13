#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import cv2
import argparse
import math
from tqdm import tqdm
import matplotlib.pyplot as plt



''' computing the Fundamental Matrix for the chess Dataset'''

def fundamental_matrix(feature_matches):
    # set parameters
    normalised = True
    thresh_val = 7
    
    # extract feature locations
    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]
    
    # check if there are enough matches
    if x1.shape[0] > thresh_val:
        # normalize the points if needed
        if normalised:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm, x2_norm = x1, x2
            
        # construct the matrix A
        A = np.zeros((len(x1_norm), 9))
        for i in range(0, len(x1_norm)):
            x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2, y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])
        
        # compute SVD of A and extract F from the last column of VT
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1].reshape(3, 3)
        
        # enforce rank 2 constraint on F
        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))
        
        # denormalize the fundamental matrix if needed
        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
        return F
    
    else:
        # return None if there are not enough matches
        return None



'''computing Essential Matrix'''

def Essential_matrix(K1, K2, F):
    # Compute essential matrix E from fundamental matrix F and camera intrinsics K1 and K2
    E = K2.T @ F @ K1

    # Apply rank constraint to enforce det(E) = 0 and preserve 5 degrees of freedom
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0
    E = U @ np.diag(S) @ Vt

    return E


'''Converting SIFT features to an array'''

def features_to_array(sift_matches, kp1, kp2):
    matching_pairs = [[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in sift_matches]
    return np.array(matching_pairs)

'''Getting x value from a line equation y=mx+c'''

def get_X(line, y):
    a= line[0]
    b = line[1]
    c = line[2]
    x = -(b*y + c)/a
    return x

'''Normalizing value of xy'''

def normalize(xy):
    # Calculate the mean of the x and y coordinates of the points
    xy_new = np.mean(xy, axis=0)
    x_new ,y_dash = xy_new[0], xy_new[1]

    # Subtract the mean from each coordinate
    x_hat = xy[:,0] - x_new
    y_hat = xy[:,1] - y_dash
    
    # Calculate the average distance from the mean
    dist = np.mean(np.sqrt(x_hat**2 + y_hat**2))
    # Scale the points so that the average distance from the mean is sqrt(2)
    s = (2/dist)
    
    # Create the scaling and translation matrices
    scaling = np.diag([s,s,1])
    T_trans = np.array([[1,0,-x_new],[0,1,-y_dash],[0,0,1]])
    T = scaling.dot(T_trans)

    # Apply the transformation to the points
    x_ = np.column_stack((xy, np.ones(len(xy))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T



'''Function to compute epipolar lines'''

def Epipolar_lines(pts_set1, pts_set2, F, image0, image1, filename, rectified=False):
    # Initialize variables
    lines1, lines2 = [], []
    epipolar_cam_0 = image0.copy()
    epipolar_cam_1 = image1.copy()
    
    # Iterate over point pairs
    for i in range(pts_set1.shape[0]):
        # Compute epipolar lines for each point pair
        line2 = np.dot(F, np.array([pts_set1[i,0], pts_set1[i,1], 1]).reshape(3,1))
        line1 = np.dot(F.T, np.array([pts_set2[i,0], pts_set2[i,1], 1]).reshape(3,1))
        lines2.append(line2)
        lines1.append(line1)
        
        # Compute endpoints of epipolar lines
        if not rectified:
            y2_min, y2_max = 0, image1.shape[0]
            x2_min, x2_max = get_X(line2, y2_min), get_X(line2, y2_max)
            y1_min, y1_max = 0, image0.shape[0]
            x1_min, x1_max = get_X(line1, y1_min), get_X(line1, y1_max)
        else:
            x2_min, x2_max = 0, image1.shape[1] - 1
            y2_min, y2_max = -line2[2]/line2[1], -line2[2]/line2[1]
            x1_min, x1_max = 0, image0.shape[1] - 1
            y1_min, y1_max = -line1[2]/line1[1], -line1[2]/line1[1]
        
        # Draw circles and lines on images
        cv2.circle(epipolar_cam_1, (int(pts_set2[i,0]), int(pts_set2[i,1])), 10, (0,0,255), -1)
        cv2.circle(epipolar_cam_0, (int(pts_set1[i,0]), int(pts_set1[i,1])), 10, (0,0,255), -1)
        epipolar_cam_1 = cv2.line(epipolar_cam_1, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (0, 255, 0), 2)
        epipolar_cam_0 = cv2.line(epipolar_cam_0, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (0, 255, 0), 2)
        
    # Resize images to same dimensions
    images_resized = [cv2.resize(img, (1920, 660)) for img in [epipolar_cam_0, epipolar_cam_1]]
    
    # Combine images and save to file
    img_group = np.concatenate(images_resized, axis=1)
    plt.imshow(img_group)
    plt.savefig(filename)
    
    return lines1, lines2



'''calculating the absolute error between the corresponding features'''

def epipolar_constraint(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1tmp=np.array([x1[0], x1[1], 1]).T
    x2tmp=np.array([x2[0], x2[1], 1])

    error = np.dot(x1tmp, np.dot(F, x2tmp))
    
    return np.abs(error)

'''RANSAC''' 

def RANSAC(features):
    # Define parameters
    iterations = 1000
    threshold_value = 0.02
    best_fit_pts = []  # Store indices of best inliers
    f_matrix = 0
    
    # Perform RANSAC iterations
    for i in range(iterations):
        # Randomly select 8 feature correspondences
        random_indices = np.random.choice(features.shape[0], size=8, replace=False)
        best_correspondences = features[random_indices, :]
        
        # Compute fundamental matrix for the 8 correspondences
        f_8 = fundamental_matrix(best_correspondences)
        
        # Find inliers using epipolar constraint
        errors = np.apply_along_axis(epipolar_constraint, 1, features, f_8)
        inlier_indices = np.where(errors < threshold_value)[0]
        
        # If number of inliers is greater than best so far, update best fit points
        if len(inlier_indices) > len(best_fit_pts):
            best_fit_pts = inlier_indices
            f_matrix = f_8
    
    # Select the best set of inliers
    required_features = features[best_fit_pts, :]
    return f_matrix, required_features

''' checking the cheirality condition to Recover the pose'''
def Cheirality(pts_3D, trans, Rot):
    return sum((Rot.dot(P.reshape(-1, 1) - trans) > 0 and P[2] > 0) for P in pts_3D)

'''restoring the camera pose '''

def Restore_camerapose(E):
    U, _, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Rot = [np.dot(U, np.dot(W * s, V_T)) for s in [1, 1, -1, -1]]
    trans = [u[:, 2] for u in [U, U, U, U]]
    signs = [1 if np.linalg.det(r) > 0 else -1 for r in Rot]
    Rot = [r * s for r, s in zip(Rot, signs)]
    trans = [t * s for t, s in zip(trans, signs)]
    return Rot, trans







def main():
    
#$$$$$$$$$$$$$$$$$$$$$$$$'''-----PART_1--- CALIBRATION --------------------'''$$$$$$$$$$$$$$$$$$$$$$$$$$$  


    # Import the SIFT algorithm from OpenCV
    sift = cv2.SIFT_create()

    # Load two chessboard images
    cam_0 = cv2.imread('data/artroom/im0.png')
    cam_1 = cv2.imread('data/artroom/im1.png')

    # Set the two images to be compared
    image0 = cam_0
    image1 = cam_1
    
    K1 = np.array([[1733.74, 0 ,792.27],[ 0, 1733.74, 541.89],[ 0 ,0, 1]])
    K2 = np.array([[1733.74, 0 ,792.27],[ 0, 1733.74, 541.89],[ 0 ,0, 1]])
    baseline = 536.62
    f = K1[0,0]
    depth_thresh = 1000000

    # Convert the two images to grayscale and RGB
    image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB) 
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Detect the SIFT features and compute their descriptors for the two images
    kp1, des1 = sift.detectAndCompute(image0_gray, None)
    kp2, des2 = sift.detectAndCompute(image1_gray, None)

    
    # Perform brute force matching
    matcher = cv2.BFMatcher()
    matches = matcher.match(des1, des2)

    # Sort matches by their distance
    matches = sorted(matches, key=lambda match: match.distance)

    # Select the best matches
    chosen_matches = matches[:100]

    print(f"Found {len(matches)} matches and selected {len(chosen_matches)} of the best ones\n")

    
    matched_image = cv2.drawMatches(image0_rgb,kp1,image1_rgb,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_image)
    plt.savefig('output/chess_matched_image.png')
    
    matched_pairs = features_to_array(chosen_matches, kp1, kp2)
   
    
    # Computing Fundamental Matrix 
    F_best, best_points = RANSAC(matched_pairs)
    

    #Computing Essential Matrix
    
    E = Essential_matrix(K1, K2, F_best)
    print('The Estimated Fundamental Matrix: \n',F_best,'\n')
    print('The Estimated Essential Matrix: \n', E,'\n')
    
    R2_, C2_ = Restore_camerapose(E)
    
    Pts_3D = []
    R1  = np.identity(3)
    C1  = np.zeros((3, 1))
    I = np.identity(3)

    # Triangulation
    for i in range(len(R2_)):
        R2 =  R2_[i]
        C2 =   C2_[i].reshape(3,1)
        Proj_mat_im0 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
        Proj_mat_im1 = np.dot(K2, np.dot(R2, np.hstack((I, -C2.reshape(3,1)))))

        for x_left_img,x_right_img in zip(best_points[:,0:2], best_points[:,2:4]):

            pts_3d = cv2.triangulatePoints(Proj_mat_im0, Proj_mat_im1, np.float32(x_left_img), np.float32(x_right_img))
            pts_3d = np.array(pts_3d)
            pts_3d = pts_3d[0:3,0]
            Pts_3D.append(pts_3d) 

    new_i = 0
    max_Positive = 0

    ### --Camera Pose Restoration--###
    for i in range(len(R2_)):
        R_, C_ = R2_[i],  C2_[i].reshape(-1,1)
        R_3 = R_[2].reshape(1,-1)
        num_Positive = Cheirality(Pts_3D,C_,R_3)

        if num_Positive > max_Positive:
            new_i = i
            max_Positive = num_Positive

    rotate, translate, P3D = R2_[new_i], C2_[new_i], Pts_3D[new_i]

    print(" Rotation matrix \n",rotate,'\n')
    print(" Translation matrix \n", translate, '\n')
    
    
#$$$$$$$$$$$$$$$$$$$$$$$$'''-----PART_2--- RECTIFICATION- --------------------'''$$$$$$$$$$$$$$$$$$$$$$$$$$$

    
    # Extract the corresponding points from best_corresponding_points
    pts_set1, pts_set2 = best_points[:, 0:2], best_points[:, 2:4]

    # Compute and save the epipolar lines
    lines1, lines2 = Epipolar_lines(pts_set1, pts_set2, F_best, image0, image1, "output/epi_polar_lines_.png", False)

    # Obtain the height and width of the images
    h1, w1 = image0.shape[:2]
    h2, w2 = image1.shape[:2]

    # Rectify the images using stereoRectifyUncalibrated
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts_set1), np.float32(pts_set2), F_best, imgSize=(w1, h1))
    print("Estimated Homography Matrix H1:\n", H1,'\n')
    print("Estimated Homography Matrix H2:\n", H2,'\n')

    # Rectify the images using the computed homography matrices
    cam_0_rectified = cv2.warpPerspective(image0, H1, (w1, h1))
    cam_1_rectified = cv2.warpPerspective(image1, H2, (w2, h2))

    # Transform the corresponding points to the rectified images
    rect_1 = cv2.perspectiveTransform(pts_set1.reshape(-1, 1, 2), H1).reshape(-1, 2)
    rect_2 = cv2.perspectiveTransform(pts_set2.reshape(-1, 1, 2), H2).reshape(-1, 2)

    # Compute the rectified fundamental matrix
    H2_T_inv = np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F_best, H1_inv))

    # Compute and save the rectified epipolar lines
    lines1_rectified, lines2_rectified = Epipolar_lines(rect_1, rect_2, F_rectified, cam_0_rectified, cam_1_rectified, "output/rectified_epi_polar_lines_.png", True)

    # Resize and convert the rectified images to grayscale
    cam_0_rectified_reshaped = cv2.resize(cam_0_rectified, (int(cam_0_rectified.shape[1] / 4), int(cam_0_rectified.shape[0] / 4)))
    cam_1_rectified_reshaped = cv2.resize(cam_1_rectified, (int(cam_1_rectified.shape[1] / 4), int(cam_1_rectified.shape[0] / 4)))
    cam_0_rectified_reshaped = cv2.cvtColor(cam_0_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    cam_1_rectified_reshaped = cv2.cvtColor(cam_1_rectified_reshaped, cv2.COLOR_BGR2GRAY)

    
    

#$$$$$$$$$$$$$$$$$$$$$$$$'''-----PART_3--- CORRESPONDENCE --------------------'''$$$$$$$$$$$$$$$$$$$$$$$$$$$


    # Taking the rectified images as input to calculate the disparity
    cam_rect_left, cam_rect_right = cam_0_rectified_reshaped, cam_1_rectified_reshaped

    cam_rect_left = cam_rect_left.astype(int)
    cam_rect_right = cam_rect_right.astype(int)

    # Initialize disparity map
    height, width = cam_rect_left.shape
    disparity_map = np.zeros((height, width))

    # Set window size and calculate new image width
    window = 4
    x_new = width - (2 * window)

    # Block Matching
    for y in tqdm(range(window, height-window)):

        # Extract blocks from rectified images
        block_cam_rect_left = []
        block_cam_rect_right = []
        for x in range(window, width-window):
            block_left = cam_rect_left[y:y + window, x:x + window]
            block_cam_rect_left.append(block_left.flatten())

            block_right = cam_rect_right[y:y + window, x:x + window]
            block_cam_rect_right.append(block_right.flatten())

        # Convert blocks to numpy arrays
        block_cam_rect_left = np.array(block_cam_rect_left)
        block_cam_rect_right = np.array(block_cam_rect_right)

        # Repeat block arrays along the third axis
        block_cam_rect_left = np.repeat(block_cam_rect_left[:, :, np.newaxis], x_new, axis=2)
        block_cam_rect_right = np.repeat(block_cam_rect_right[:, :, np.newaxis], x_new, axis=2)

        # Transpose right block array
        block_cam_rect_right = block_cam_rect_right.T

        # Calculate Sum of Absolute Differences (SAD)
        absolute_difference = np.abs(block_cam_rect_left - block_cam_rect_right)
        SAD = np.sum(absolute_difference, axis=1)
        index = np.argmin(SAD, axis=0)
        disparity = np.abs(index - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity

    # Plot disparity maps
    print('Plotting disparity maps ')
    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))
    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    plt.savefig('output/disparity_image_heat' + ".png")
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('output/disparity_image_gray' + ".png")
    

#$$$$$$$$$$$$$$$$$$$$$$$$'''-----PART_4--- DEPTH --------------------'''$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    # Thresholding depth values
    depth = (baseline * f) / (disparity_map + 1e-10)
    depth[depth > depth_thresh] = depth_thresh

    # Plotting Depth Maps
    print('Plotting Depth maps')
    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('output/depth_image_heat' + ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('output/depth_image_gray' + ".png")
    plt.show()
    
if __name__ == '__main__':
    main()





# In[ ]:




