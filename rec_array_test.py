import urllib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# grab the price data off yahoo
u1 = urllib.urlretrieve('http://ichart.finance.yahoo.com/table.csv?s=AAPL&d=9&e=14&f=2008&g=d&a=8&b=7&c=1984&ignore=.csv')
u2 = urllib.urlretrieve('http://ichart.finance.yahoo.com/table.csv?s=GOOG&d=9&e=14&f=2008&g=d&a=8&b=7&c=1984&ignore=.csv')

# load the CSV files into record arrays
r1 = mlab.csv2rec(file(u1[0]))
r2 = mlab.csv2rec(file(u2[0]))

# compute the daily returns and add these columns to the arrays
gains1 = np.zeros_like(r1.adj_close)
gains2 = np.zeros_like(r2.adj_close)
gains1[1:] = np.diff(r1.adj_close)/r1.adj_close[:-1]
gains2[1:] = np.diff(r2.adj_close)/r2.adj_close[:-1]
r1 = mlab.rec_append_fields(r1, 'gains', gains1)
r2 = mlab.rec_append_fields(r2, 'gains', gains2)

# now join them by date; the default postfixes are 1 and 2
r = mlab.rec_join('date', r1, r2)

# long appl, short goog
g = r.gains1-r.gains2
tr = (1+g).cumprod()  # the total return

# plot the return
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r.date, tr)
ax.set_title('total return: long appl, short goog')
ax.grid()
fig.autofmt_xdate()
plt.show()