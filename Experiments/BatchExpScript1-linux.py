

import sys

work_dir = '/home/pgrad/musselle/linux/FRAHST_Project/'

sys.path.append( work_dir + 'Algorithms')
sys.path.append( work_dir + 'CMA')
sys.path.append( work_dir + 'Experiments')
sys.path.append( work_dir + 'Utils')

execfile( work_dir + 'Experiments/gen_anom_batch_exp1-shift_up_down-l.py')
execfile( work_dir + 'Experiments/gen_anom_batch_exp2-step-l.py')
execfile( work_dir + 'Experiments/gen_anom_batch_exp3-step_n_back-l.py')
execfile( work_dir + 'Experiments/gen_anom_batch_exp4-peak_dip-l.py')