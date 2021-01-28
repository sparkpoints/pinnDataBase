# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from scipy.interpolate import griddata
from  dicompylercore import dicomparser,dose
import numpy as np

# %%
doseObj = dicomparser.DicomParser('RD.dcm')
doseM = dose.DoseGrid(doseObj.ds)
doseData = np.asarray(doseM.dose_grid)
print(type(doseData),doseData.shape)
f,l,c = doseData.shape
# %%
f = f*4
l = l*4
c = l*4
newDose = doseData.reshape([f,l,c])
# rsOjb = dicomparser.DicomParser('RS.dcm')

