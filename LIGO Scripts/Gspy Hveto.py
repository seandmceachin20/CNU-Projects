#"""Download CSV file of triggers, then format command to run Hveto on triggers"""
#some code copied from Joe Areeda https://git.ligo.org/joseph-areeda/hvetoSupport/-/blob/master/bin/gspy_get.py
#use environment: conda activate igwn-py39-20221118
from gwpy.table import EventTable
from gwpy.time import tconvert
from glue.lal import LIGOTimeGPS
import numpy as np
from glue.ligolw import utils, ligolw, lsctables
from pathlib import Path
import sys

from astropy.time import Time

from gwpy.segments import DataQualityFlag
from gwpy.segments import SegmentList
from gwpy.segments import Segment

import h5py
import os

# %%
input = sys.argv
start = tconvert(str(input[1]))
end = tconvert(str(input[2]))
ifo = 'L1'
glitch = str(input[3])

# %%
output_path = "output_gspy/" + str(start)+"_"+str(end)+"_"+glitch
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %%
label = "ml_label="+glitch
duration = end - start

events = EventTable.fetch('gravityspy','glitches_v2d0',
                                  selection=[f"{label}",'ml_confidence>.9',
                                             f'{start}<event_time<{end}',
                                             f'ifo={ifo}'],
                                  user='mla', passwd='REDACTED',
                                  host='gravityspyplus.ciera.northwestern.edu')

t0 = None
tlast = None
seg_start = None
psegs = SegmentList()
doWarn = True
n = 0
seg_tmin = None
seg_tmax = None
pad = 100

sngl_burst = lsctables.New(lsctables.SnglBurstTable, ['peak_time','peak_time_ns','peak_frequency','snr'])

for t, f, s in zip(map(LIGOTimeGPS,events['event_time']),events['peak_frequency'],events['snr']):
    row = sngl_burst.RowType()
    row.set_peak(t)
    row.peak_frequency = f
    row.snr = s
    sngl_burst.append(row)

output_file = 'L1-GDS_CALIB_STRAIN_blips-{}-{}.xml.gz'.format(start, end)
xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
xmldoc.childNodes[0].appendChild(sngl_burst)
utils.write_filename(xmldoc, "temp_gspy//" + output_file, gz=True)

# %%
cache = open("cache.lcf", "w")
cache.write(str(Path.cwd()) + "/temp_gspy/" + output_file)


# %%
print("conda activate /home/derek.davis/.conda/envs/hveto_dev")
print("hveto " + str(start) + " " + str(end) + " --ifo L1 --config-file /home/sean.mceachin/public_html/gspy_hveto/O3config.ini --primary-cache cache.lcf --output-directory " + output_path + " --nproc 4".format(start, end))

