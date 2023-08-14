# SuperImposing Images on AprilTag - 2D &3D [AR Application]

## AprilTag Detection

- Performed FFT on image to isolate tag from background  
- Detected tag corners with Shi-Tomasi corner detector
- Computed homography between tag and reference to find ID and orientation
- Implemented grid encoding scheme to decode tag ID from inner pixel values

## AprilTag Tracking

- Computed homography between tag and Testudo logo image
- Warped Testudo image to overlay on tag, keeping consistent orientation
- Calculated projection matrix from camera intrinsics and tag homography
- Projected 3D cube corners using projection matrix for virtual overlay 

## Methodology

- Used Shi-Tomasi corner detection and FFT for tag isolation
- Custom warpPerspective implementation for homography transforms 
- Developed grid encoding scheme to extract ID from tag image

## Results

- Successfully detected and decoded ID of AprilTag from video frame
- Overlaid Testudo logo by warping via tag-image homography
- Achieved virtual 3D cube overlay using computed projection matrix

## Discussion

- Homography estimation enables overlaying virtual objects on detected tags
- Projection matrix allows realistic rendering with viewpoint changes
- Future work on making overlays more robust to lighting and occlusion

## References

[1] Project Report

Let me know if you would like me to expand or modify this computer vision README.
