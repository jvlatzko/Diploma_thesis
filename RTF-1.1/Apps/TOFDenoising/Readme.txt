This example implements a somewhat simplified implementation of the
natural image denoising model described in:

J. Jancsary, S. Nowozin, and C. Rother.  Loss-Specific Training of
Non-Parametric Image Restoration Models: A New State of the Art.  In
12th European Conference on Computer Vision (ECCV), Florence, Italy,
2012.

It is a good starting point for any kind of low-level image processing
application and demonstrates the following features:

- Loss-based training of model parameters and tree structure

- Stacked training of RTF model (i.e. a cascade of RTF models, each
  of which is based on the prediction of the previous model)

- Serialization and deserialization of model files from/to disk

- Parallelizing training over several machines using MPI
