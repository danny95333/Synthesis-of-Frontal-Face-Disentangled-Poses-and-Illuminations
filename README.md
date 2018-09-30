# Siamese-Network-for-Frontal-Face-Synthesis-disentangle-pose-and-light-interference-

All the input data are from MultiPIE dataset, 9 different identity faces each time with different poses and illumination.
![input](https://github.com/danny95333/Synthesis-of-Frontal-Face-Disentangled-Poses-and-Illuminations/blob/master/input_samples_iteration_200.png)

## First version: complete basic functions, the frontal face image has clear identity, but blurry boundary and glasses still can't be 100% synthesis.
>siamese_out_largeP&LclassW: these are the output with relative large class_P and class_L LOSS.
![first_L](https://github.com/danny95333/Synthesis-of-Frontal-Face-Disentangled-Poses-and-Illuminations/blob/master/siamese_out_largeP%26LclassW/fake_samples_iteration_70000.png)
>siamese_out_sameW: these are the output with same weigths to every LOSSes.
![first_Same](https://github.com/danny95333/Synthesis-of-Frontal-Face-Disentangled-Poses-and-Illuminations/blob/master/siamese_out_sameW/fake_samples_iteration_70000.png)
## 2nd update at 30th June
* load from June-20 pretrained model and get rid of other losses but L1 loss
>2nd_step_only_L1: these are the 2nd update's output with 'Step_decay_learning_rate'
![second_l1](https://github.com/danny95333/Synthesis-of-Frontal-Face-Disentangled-Poses-and-Illuminations/blob/master/2nd_step_only_L1/fake_samples_iteration_40000.png)
## 3rd update: Mask was sample from 20 frontal face image manually, the model and weights was based on the 1st version.
* Add Mask at frontal face's boundary to make the netS be more sensitive to the boundary region
* Freeze the pre-trained Siamese net's encoder part, which aimed to capture the input images' feature maps. And train the decoder part of the network.
>3rd_step_L1_Mask_FreezEnc: These are the 3rd update's output with 'Step_decay_learning_rate', still using L1 as general supervise loss.
![third_l1](https://github.com/danny95333/Synthesis-of-Frontal-Face-Disentangled-Poses-and-Illuminations/blob/master/3rd_step_L1_Mask_FreezEnc/fake_samples_iteration_30000.png)
>3rd_step_L2_Mask_FreezEnc: These are the 3rd update's output with 'Step_decay_learning_rate', still using L2 as general supervise loss.
Problem: There are still many artifacts on the synthesized images, not 'real' enough 
![third_l2](https://github.com/danny95333/Synthesis-of-Frontal-Face-Disentangled-Poses-and-Illuminations/blob/master/3rd_step_L2_Mask_FreezEnc/fake_samples_iteration_30000.png)



