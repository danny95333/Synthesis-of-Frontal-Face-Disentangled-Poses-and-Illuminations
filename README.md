# Siamese-Network-for-Frontal-Face-Synthesis-disentangle-pose-and-light-interference-
Based on the paper "Deep Disentangling Siamese Network for Frontal Face Synthesis under Neutral Illumination" by Ting Zhang

First version: complete basic functions, the frontal face image has clear identity, but blurry boundary and glasses still can't be 100% synthesis.

# 2nd update at 30th June
1. Add Mask at frontal face's boundary to make the netS be more sensitive to the boundary region
2. Freeze the pre-trained Siamese net's encoder part, which aimed to capture the input images' feature maps. And train the decoder part of the network.

Mask was sample from 20 frontal face image manually, the model and weights was based on the 1st version.

Problem: There are still many artifacts on the synthesized images, not 'real' enough 
# 3rd update will be updated at 3rd July(add GAN for several different attempts)

