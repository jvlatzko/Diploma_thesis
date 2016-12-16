This source code release implements various training algorithms for
"Regression Tree Fields", described in the following papers:

J. Jancsary, S. Nowozin, T. Sharp, and C. Rother.  Regression Tree
Fields – An Efficient, Non-Parametric Approach to Image Labeling
Problems.  In 2012 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), Providence, RI, USA, 2012.

J. Jancsary, S. Nowozin, and C. Rother.  Loss-Specific Training of
Non-Parametric Image Restoration Models: A New State of the Art.  In
12th European Conference on Computer Vision (ECCV), Florence, Italy,
2012.

J. Jancsary, S. Nowozin, and C. Rother.  Learning Convex QP
Relaxations for Structured Prediction.  In 30th International
Conference on Machine Learning (ICML), Atlanta, GA, USA, 2013.

U. Schmidt, C. Rother, S. Nowozin, J. Jancsary, and S. Roth.
Discriminative Non-blind Deblurring.  In 2013 IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), Portland, OR, 2013.

Folders
=======

  ./RTF     ... contains the header-only RTF library
  ./Apps    ... contains example applications
  ./Data    ... contains small datasets for the examples
  ./Models  ... contains learned RTF models


Building
========

The RTF code is header-only and as such it need not be built. However,
you may wish to build the example applications. To do so, please
follow the instructions in either BuildingOnWindows.txt or
BuildingOnUnix.txt.


Using
=====

To build your own application based on the RTF framework, it is best
to start out from one of the example applications. Typically, an
application will need to implement the following classes:

  - A dataset class, which is responsible for reading in the training
    and test data, and which provides that data to the RTF algorithms.

  - A feature class, which is responsible for extracting features from
    an input instance; this includes the feature checks that are
    performed at each internal node of the regression tree, as well as
    the basis vectors that are used in the leaf models.

  - A feature sampler class which is responsible for sampling from the
    space of all possible features. That is, the feature sampler is 
    expected to instantiate a user-specified number of features, which
    will then be evaluated regarding their utility in discriminating
    the data by the tree growing algorithm.

These classes are passed to the training algorithms as template
parameters. As such, the classes are expected to implement a certain
interface. The RTF code base draws heavily on compile-time
polymorphism: The dataset and feature classes need not be derived from
a base class; instead, they are only required to contain
implementations of the methods that are actually invoked from the C++
template code.

Again, the easiest way of implementing suitable dataset and feature
classes is to start out from one of the examples in the Apps
directory.

The ensure all the right compiler and linker flags are set, the
easiest route is create a new directory for your own application in
the Apps folder, modify the Apps/CMakeLists.txt file to include that
directory, and to copy the CMakeLists.txt file of an example
application into your own application directory. You can then simply
build your application in the existing RTF source tree.

For many purposes, the simple API exposed in RTF/Basic.h or
RTF/Stacked.h should be sufficient, so these two headers are often a
good starting point to become familiar with the RTF API. These two
files contain the majority of the API documentation that is relevant
to a typical user.

---
Last modified by Uwe Schmidt on 07/05/2014.