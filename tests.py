'''
Things to test:

1) L2 reconstruction loss. Did I actually code it up properly? It can be confusing with reduce mean and reduce sum over different axes
2) MMD. I have never used tf.add_n() before. Does this correctly add up the separate MMDs?
3) total_loss sum over separate losses using tf.add_n()
'''
