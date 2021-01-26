from typing import Pattern
import matplotlib.pyplot as plt
from dicompylercore import dicomparser, dvh, dvhcalc
from glob import glob
import os,re
# from .PyrexReader import Img_Bimask
import SimpleITK as sitk
import numpy as np


rs = ''
rp = ''
rd = ''

path = '/Users/yang/Downloads/ESO/120115'
# fileL = glob('/Users/yang/Downloads/ESO/120115/*.dcm')
# pattern = '%s/*.dcm'%path

# fileList = glob(os.path.join(path,'*.dcm'))
# print(fileList)
pattern = '%s/*.dcm'%path

fileList = glob(os.path.join(path,'*.dcm'))
# print(fileList)

for file in os.listdir(path):
    if 'RS' in file.upper():
	    rs = os.path.join(path, file)
    if 'RP' in file.upper():
	    rp = os.path.join(path, file)
    if 'RD' in file.upper():
	    rd = os.path.join(path, file)

print(rs,rp,rd)

dp = dicomparser.DicomParser(rs)
print(dp.GetStructures()[5])

dose = dicomparser.DicomParser(rd)
rawData = dose.GetDoseData()['lut']

print(type(rawData))
doseM = np.array(rawData,dtype=object)
print(type(doseM))
print(len(doseM))
# dosemesh = sitk.GetImageFromArray(dose)
# dvh = dvh.DVH.from_dicom_dvh(dose.ds,5)

# print(dvh.describe())


# calcDVH = dvhcalc.get_dvh(rs,rd,5)
# print(calcDVH.describe())
# print("maxdose:%s,mindose:%s,D2cc:%s"%(calcDVH.max,calcDVH.min,calcDVH.D2cc))
# calcDVH.plot()
# plt.show()

# Img,Mask = Img_Bimask(path,path,'PTV')
# Img.shape(
