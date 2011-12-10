
from pylab import *
from plot_utils import plot_2x1

# Must Define 
#   data2plot
#   EH
#   EL

#ylims = (2,0.92,1.0)

# Title = r'$SNR = -9$'

plot_2x1(shiftStreams, peakStreams, 
         ['Signal Level', 'Signal Level'], 'Time steps')

fig = gcf()

# dashed lines
fig.axes[0].axvline(x = 300, ls = '--', c = 'k')
fig.axes[0].axvline(x = 600, ls = '--', c = 'k')
fig.axes[1].axvline(x = 300, ls = '--', c = 'k')
fig.axes[1].axvline(x = 600, ls = '--', c = 'k')

#fig.axes[1].axhline(y=EH, ls = '--')
#fig.axes[1].axhline(y=EL, ls = '--')

fig.show()

# Save figure as eps

filename  = 'peak&shift inputs'
artistList = fig.axes[0].texts + fig.texts
#artistList = fig.axes[0].texts + fig.
fig.savefig(filename + '.eps', bbox_inches = 'tight', 
            bbox_extra_artists = artistList )


