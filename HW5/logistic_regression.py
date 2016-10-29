import numpy as np

""""
Methods to do logistic regression in the 2D plane from [-1,1]x[-1,1].
Error will be measured using the "cross entropy error"  function ln(1 +e^(-(y_n)transpose(w)x_n) at each time step
At each epoch the reg will go through each xi, yi pair in a random order.
The alg will stop when the delta w is below 0.01 (measured at the end of an epoch)
"""

