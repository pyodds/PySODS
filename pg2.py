import time
import numpy as np

start_time = time.clock()
for i in range(100):
    np.dot(np.random.rand(400,400),np.random.rand(400,400))
print('Total cost: %.6f seconds' % (time.clock() - start_time))
