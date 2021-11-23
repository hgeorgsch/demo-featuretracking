# demo-featuretracking

This projects is intended to demonstrate 
corner detectection and tracking by means
of signal differentiation.

There are two demo scripts in the src/ directory.

+ `detector.py` runs the Harris Corner Detector
  and annotates the image showing red circles 
  around the feature points. 
+ `tracker.py` attempts to calculate the motion
  vectors and annotate the frame with both new
  and old feature points and motion vectors.

The former works satisfactorily, although it 
detects many weak and useless features.  
No thresholding is used.

The latter does not work.  The main reason for
this is probably that multi-scale tracking is 
not applied.  This should be added for a useful
test.
