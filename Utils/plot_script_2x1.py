
from pylab import *
from plot_utils import plot_2x1

# Run this script interactively:
# Must first Define 
#   data2plot
#   EH
#   EL

ylims = (2,0.91,1.0)

Title = ''

plot_2x1(F.res['ht'], F.res['e_ratio'], 
         ['Hidden Variables', 'Energy Ratio'], 'Time Steps',Title,  ylims= ylims)

fig = gcf()

# dashed lines
fig.axes[0].axvline(x = 300, ls = '--', c = 'k')
fig.axes[0].axvline(x = 600, ls = '--', c = 'k')
fig.axes[1].axvline(x = 300, ls = '--', c = 'k')
fig.axes[1].axvline(x = 600, ls = '--', c = 'k')

fig.axes[1].axhline(y=EH, ls = '--')
fig.axes[1].axhline(y=EL, ls = '--')

fig.show()

# Save figure as eps

filename  = 'FRAHST shift baselin 0 - new'
artistList = fig.axes[0].texts 
#artistList = fig.axes[0].texts + fig.texts

fig.savefig(filename + '.eps', bbox_inches = 'tight', 
            bbox_extra_artists = artistList )


