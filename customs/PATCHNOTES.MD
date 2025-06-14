# Hi these are some patchnotes for stuff i had to do to get DreamerV3 working

For customs/dreamerv3_tf_learner.py: 
- reduce=None for any call self.metrics.log_dict() with '_L_TOTAL' as a logging string.