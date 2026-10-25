# Face Recognition & Verification System (Siamese Networks)

This project is part of the Deep Learning Specialization, focusing on face verification and face recognition using **Siamese Networks** and **Triplet Loss**. The system learns a function that maps an image into a 128-dimensional embedding space, enabling one-shot learning and accurate person identification.

## Project Overview & Core Components

1. **Triplet Loss (`triplet_loss`)**:
   - Implements the custom loss function used to train the Siamese network.
   - Minimizes the distance between an anchor image ($A$) and a positive image ($P$) of the same person, while maximizing the distance to a negative image ($N$) of a different person, incorporating a margin $\alpha$.

2. **Face Verification (`verify`)**:
   - Performs a 1-to-1 comparison to check whether an incoming door-camera image matches a claimed identity.
   - Computes the L2 norm distance between the image encoding and the database reference vector against a predefined threshold (0.7).

3. **Face Recognition (`who_is_it`)**:
   - Performs a 1-to-K identification task without requiring a claimed identity beforehand.
   - Loops through the authorized personnel database, calculates L2 distances to find the closest match, and verifies if the person is within the authorized threshold.

## Required Libraries

- TensorFlow (v2.x)
- NumPy
- Keras
- Matplotlib
- PIL (Pillow)
