The input should be 4 channels of a 256x256 image representing the fluid.

  - Split the domain into 4*4 patches. These represent the image level. These patch tokens have low dimensionality.
  - Above this, we have a feature hierarchy of higher level tokens with much higher dimensionality. There should be a pyramid where we have 32x32, 64x64, 128x128 and 256x256 (with
  the entire image being 256x256).
  - There should be two copies of this hierarchy. The first should contain the feature map which absorbes higher level structural featues of the image. The second is the system 
  embedding which contains constant information about the fluid being simulated/boundary conditions etc. This can be of lower dimensionality.
  - Finally, there is a second residual layer that can be fed into the model, with the same 4*4 size and embeddings. This will take in prediction error values and is used to calibrate the
  system embeddings.
  - Attention should be as follows.
    - Each image level patch has local attention to nearby image layer patches (perhaps within a 3x3 radius) and to the local feature embeddings.
    - Each residual layer has local attention over only residual layer patches and to the local system embeddings.
    - Each feature map token has full attention over all other feature map embeddings, all other system embeddings and over the patches in its region.
    - The system embedding tokens have attention oveer all other system embeddings, feature map embeddings, and local residual patches. Although these values can only be changed if the residual layer is in the system. 
  - At the end of the network we have a Generative Adversarial Network (GAN) discriminator. Its job is to predict whether the output is real or fake and adapt accordingly. This exists to ensure that the output of the model actually produces something that is reasonable. Both the final decoded frame and the the higher level feature tokens should be included (but the feature tokens should not have their values fixed - they should not be updated directly here).

The training loop works as follows:

  - First start with a number of ground truth frames. We initialise zeroed system embeddings because we don't know how the system behaves initially.
  - We pass through the initial frames into the model.
  - We also pass through the residual frames after each output. We do this for 5 frames say. At the same time, the system embedding is developed.
  - At some point we then run an actual training run. When we do this, the system tokens are frozen so they do not change throughout the transformer. When we backprop, we still backprop through the earlier initial runs which led to these system tokens being as they are.
