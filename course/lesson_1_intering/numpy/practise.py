import numpy as np
import time

def func():
  start = time.time()
  A = np.random.randint(-10,10,size=(5_000,5_000))
  print(A)
  print()
  print(A<4)
  print("\n")
  A_masked = A[:,(A>0).sum(axis=0) > (A<0).sum(axis=0)]

  print(A_masked)
  print(time.time() - start)
func()