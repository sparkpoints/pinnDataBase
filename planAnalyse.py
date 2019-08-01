####################################################################################################################################################
#   import libraries below
####################################################################################################################################################
from __future__ import print_function

import operator
# import pydicom.uid
import argparse
import os
import sys
import os.path
import re  # used for isolated values from strings
import struct
import time  # used for getting current date and time for file
from functools import reduce
from random import randint

from glob import glob
import matplotlib.pyplot as plt
from pymedphys.gamma import gamma_dicom, gamma_percent_pass, gamma_filter_numpy
from pymedphys.dicom import zyx_and_dose_from_dataset
from pymedphys.pinnacle import PinnacleExport
from pymedphys_pinnacle.export import pinnacle_cli

import numpy as np
import pydicom as dicom
import pydicom.uid
from dicompylercore import dicomparser, dvhcalc
from dicompylercore.dvh import DVH
from pydicom.dataset import Dataset, FileDataset
from pydicom.filebase import DicomFile
from pydicom.sequence import Sequence

from pinn2Json import pinn2Json

# import logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     filename='convert_whole_patient.log',
#                     filemode='w')
# # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
#
# debugmode = True


def getTPSDCM(tpsDVHHome, patienMRN):
    Rs = None
    Rd = None
    dirList = os.listdir(tpsDVHHome)
    for dvh in dirList:
        if dvh == patienMRN:
            for dcm in os.listdir(os.path.join(tpsDVHHome, dvh)):
                logging.info(dcm)
                if 'RS' in dcm:
                    Rs = pydicom.read_file(os.path.join(tpsDVHHome, dvh, dcm))
                    Rs = dicomparser.DicomParser(Rs)
                if 'RD' in dcm:
                    Rd = pydicom.read_file(os.path.join(tpsDVHHome, dvh, dcm))
                    Rd = dicomparser.DicomParser(Rd)
        if Rs and Rd:
            break
    return Rs, Rd
class arges():
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_directory = output_path
        self.verbose = False
        self.modality = ["CT", "RTSTRUCT", "RTDOSE", "RTPLAN"]
        self.plan = ''
        self.trial = ''
        self.list = ''
        self.image = ''
        self.uid_prefix = ''

if __name__ == "__main__":
    #arg1,arg2,arg3 = sys.argv[1:]
    workingPath = ''
    inputfolder = ''
    outputfolder = ''
    tpsDVHsDir = ''
    if sys.platform == 'linux' or sys.platform == 'linux2':
        workingPath = '/home/peter/PinnWork'
        inputfolder = os.path.join(workingPath, 'Accuracy', 'Mount_CIRS/')
        outputfolder = os.path.join(workingPath, 'export_dicom_pool/')
        tpsDVHsDir = os.path.join(workingPath, 'Accuracy', 'tps_dcm_CIRS')
    elif sys.platform == 'darwin':
        workingPath = '/Users/yang/PinnWork'
        inputfolder = os.path.join(workingPath, 'Accuracy', 'CIRS_Raw/')
        outputfolder = os.path.join(workingPath, 'Accuracy', 'export/')
        tpsDVHsDir = os.path.join(workingPath, 'Accuracy', 'CIRS_Tps/')
    elif sys.platform == 'win32':
        pass

    # log file
    logfile = os.path.join(workingPath, 'runlogger', time.strftime(
        "%Y%m%d-%H%M%S") + 'dvhdata.csv')

    # analysePlan(os.path.join(inputfolder, 'Patient_38703'),
    #             outputfolder, logfile)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_path',)
    pinnObject = pinn2Json()
    patientList = os.listdir(inputfolder)
    for patient in patientList:
        if os.path.isfile(os.path.join(inputfolder, patient, 'Patient')):
            patientInfo = pinnObject.read(os.path.join(inputfolder, patient, 'Patient'))
            print(patientInfo.PatientID, patientInfo.MedicalRecordNumber,
                  (patientInfo.FirstName + patientInfo.LastName))
            input_path = os.path.join(inputfolder, patient)
            output_path = os.path.join(outputfolder,patientInfo.MedicalRecordNumber)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            argobj = arges(input_path,output_path)
            pinnacle_cli.export_cli(argobj)


