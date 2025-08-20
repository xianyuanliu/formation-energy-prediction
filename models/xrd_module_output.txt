1. Loading data using XRDDataset...
   ...Successfully loaded 4471 samples.
   ...Detected 128 features per sample.
2. Creating an instance of the XRDFeatureExtractor model.
   XRDFeatureExtractor(
   (extractor): Sequential(
   (0): Linear(in_features=128, out_features=256, bias=True)
   (1): ReLU()
   (2): Linear(in_features=256, out_features=128, bias=True)
   (3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
   )
   )
3. Creating a DataLoader to batch the real data.
4. Passing a batch of real data through the model.

--- Test Results ---
Input Data Shape: torch.Size([32, 128])
Output Data Shape: torch.Size([32, 128])

✅ Test Successful: Model processed the data and produced the correct output shape.
