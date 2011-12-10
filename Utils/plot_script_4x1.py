
from pylab import *
from plot_utils import plot_4x1

# Must Define 
#   data2plot
#   EH
#   EL

ylims = (4,0.92,1.0)

# Title = r'$SNR = -9$'


plot_4x1(streams, results[0]['data']['hidden'], results[0]['data']['r_hist'], results[0]['data']['e_ratio'],
         ['Input Data', 'Hidden\nVariables', 'Rank', 'Energy\nRatio'], 'Time steps', ylims = ylims)

fig = gcf()

# dashed lines
fig.axes[0].axvline(x = 300, ls = '--', c = 'k')
fig.axes[0].axvline(x = 600, ls = '--', c = 'k')
fig.axes[1].axvline(x = 300, ls = '--', c = 'k')
fig.axes[1].axvline(x = 600, ls = '--', c = 'k')
fig.axes[2].axvline(x = 300, ls = '--', c = 'k')
fig.axes[2].axvline(x = 600, ls = '--', c = 'k')
fig.axes[3].axvline(x = 300, ls = '--', c = 'k')
fig.axes[3].axvline(x = 600, ls = '--', c = 'k')

fig.axes[3].axhline(y=EH, ls = '--')
fig.axes[3].axhline(y=EL, ls = '--')

fig.show()

# Save figure as eps

filename  = 'Res_v6.3-adaptive'
artistList = fig.axes[0].texts 
#artistList = fig.axes[0].texts + fig.
fig.savefig(filename + '.eps', bbox_inches = 'tight', 
            bbox_extra_artists = artistList )


