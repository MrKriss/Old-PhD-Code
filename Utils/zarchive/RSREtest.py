


import numpy as np 


# def data and recon


b = np.zeros(data.shape[0])
a = np.zeros(data.shape[0])
RSRE = np.zeros(data.shape[0])


for i in range(data.shape[0]):

  if i == 0:
    RSRE[i] = 0.0
    b[i] = 0.0
    a[i] = 0.0
  else:
    b[i] = b[i-1] + np.linalg.norm(data[i,:]) ** 2
    a[i] = a[i-1] + np.linalg.norm(data[i,:] - recon[i,:]) ** 2
    RSRE[i] = a[i] / b[i]
    

RSRE2 = np.zeros(data.shape[0])
alpha = 0.96

for i in range(data.shape[0]):

  if i == 0:
    RSRE2[i] = 0.0
    b[i] = 0.0
    a[i] = 0.0
  else:
    b[i] = np.linalg.norm(data[i,:]) ** 2
    a[i] = np.linalg.norm(data[i,:] - recon[i,:]) ** 2
    RSRE2[i] = (alpha * RSRE2[i-1]) + (a[i] / b[i])
    