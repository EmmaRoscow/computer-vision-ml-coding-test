This folder contains scripts to run a neural network model to predict gender from images, as well as scripts to train the neural network. It runs on PyTorch.

The model is saved in models/simple_cnn.

To run the model on the image:
1. Run cg = classify_gender() to load the model
2. Run cg.classify(image) where image is either a single image of size 28 x 28 x 3 or a batch of n images of size n x 28 x 28 x 3. For each image it will input either 1 (male) or 0 (female)

Optional: to train the model from scratch, run train_model_main()