#!/usr/bin/env python
#coding:utf-8
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False
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
        self.verbose = True
        self.modality = ["CT", "RTSTRUCT", "RTDOSE", "RTPLAN"]
        self.plan = ''
        self.trial = ''
        self.list = ''
        self.image = ''
        self.uid_prefix = ''
def CleaningPatientPool(poolpath):
    MatchRegList = ['^.auto.plan*',
                    '.ErrorLog$',
                    '.Transcript$',
                    '.defaults$',
                    '.pinnbackup$',
                    '^Institution.\d+',
                    '^Patient.\d+',
                    '\s*~\s*',
                    '.json$']

    for dirpath, dirname, filenames in os.walk(poolpath):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            if os.path.islink(filepath):
                os.remove(filepath)
                continue
            for currentReg in MatchRegList:
                if re.findall(currentReg, file):
                    os.remove(filepath)
                    print('del:%s\n' % filepath)
        if dirname:
            for currendir in dirname:
                CleaningPatientPool(os.path.join(dirpath,currendir))

def calcGamma(refRD,evaRD):
    reference = pydicom.dcmread(refRD)
    evaluation = pydicom.dcmread(evaRD)

    patientID = evaluation['PatientID'].value

    doseThresh = [1,2,3]
    dtaThresh = [1,2,3]
    gamma_options = {
        'dose_percent_threshold': 1,
        'distance_mm_threshold': 1,
        'lower_percent_dose_cutoff': 10,
        'interp_fraction': 5,
        'max_gamma': 2,
        'random_subset': None,
        'local_gamma': True,
        'ram_available': 2 ** 29
    }

    fig, ax = plt.subplots(figsize=(13, 10), nrows=len(doseThresh), ncols=len(dtaThresh))
    for dosei in doseThresh:
        for dtaj in dtaThresh:
            # subname = str('c' + dosei + dtaj)

            gamma_options['dose_percent_threshold'] = dosei
            gamma_options['distance_mm_threshold'] = dtaj

            gamma = gamma_dicom(reference, evaluation, **gamma_options)
            # %%
            valid_gamma = gamma[~np.isnan(gamma)]

            num_bins = (gamma_options['interp_fraction'] * gamma_options['max_gamma'])
            bins = np.linspace(0, gamma_options['max_gamma'], num_bins + 1)

            ax[dosei-1, dtaj-1].hist(valid_gamma, bins, density=True)
            ax[dosei-1, dtaj-1].set_xlim([0, gamma_options['max_gamma']])

            pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)

            titleName = "Gamma:{0}%,{1}mm, PassRate: {2:.2f}%".format(dosei,dtaj,pass_ratio * 100)
            ax[dosei-1, dtaj-1].set_title(titleName)
            # plt.savefig('gamma_hist.png', dpi=300)
    plotName = patientID + 'hist_gramm_grid.png'
    data_path = os.path.join(os.getenv('HOME'),'PinnWork','plotdata',patientID)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    plt.savefig(os.path.join(data_path,plotName), dpi=300)
    plt.close()

    #use 3%,2mm, cretia calc gamma for slices ploting
    gamma_options['dose_percent_threshold'] = 3
    gamma_options['distance_mm_threshold'] = 2

    gamma = gamma_dicom(reference, evaluation, **gamma_options)
    # # %%
    # valid_gamma = gamma[~np.isnan(gamma)]
    #
    # num_bins = (gamma_options['interp_fraction'] * gamma_options['max_gamma'])
    # bins = np.linspace(0, gamma_options['max_gamma'], num_bins + 1)

    # %% plot slices
    (z_ref, y_ref, x_ref), dose_reference = zyx_and_dose_from_dataset(reference)
    (z_eval, y_eval, x_eval), dose_evaluation = zyx_and_dose_from_dataset(evaluation)

    dose_reference = dose_reference * 100
    dose_evaluation = dose_evaluation * 100

    lower_dose_cutoff = gamma_options['lower_percent_dose_cutoff'] / 100 * np.max(dose_reference)

    relevant_slice = (
            np.max(dose_reference, axis=(1, 2)) >
            lower_dose_cutoff)
    slice_start = np.max([
        np.where(relevant_slice)[0][0],
        0])
    slice_end = np.min([
        np.where(relevant_slice)[0][-1],
        len(z_ref)])
    max_ref_dose = np.max(dose_reference)

    z_vals = z_ref[slice(slice_start, slice_end, 20)]

    # for z_i in z_vals:
    #     for i,z_axal in enumerate(z_eval,0):
    #         if z_i == z_axal:
    #             eval_slices.append(dose_evaluation[i,:,:])
    eval_slices = [
        dose_evaluation[np.where(z_i - z_eval < 0.01)[0][0], :, :]
        for z_i in z_vals
    ]

    ref_slices = [
        dose_reference[np.where(z_i - z_eval < 0.01)[0][0], :, :]
        for z_i in z_vals
    ]

    gamma_slices = [
        gamma[np.where(z_i - z_eval < 0.01)[0][0], :, :]
        for z_i in z_vals
    ]

    diffs = [
        eval_slice - ref_slice
        for eval_slice, ref_slice
        in zip(eval_slices, ref_slices)
    ]

    max_diff = np.max(np.abs(diffs))

    for i, (eval_slice, ref_slice, diff, gamma_slice,z_location) in enumerate(zip(eval_slices, ref_slices, diffs, gamma_slices,z_vals)):
        fig, ax = plt.subplots(figsize=(13, 10), nrows=2, ncols=2)

        #add calobar
        # cmap = colors.ListedColormap(['b','g','y','r'])
        # bounds = list(np.arange(0,max_ref_dose,max_ref_dose/5))
        # norm = colors.BoundaryNorm(bounds,cmap.N)

        c00 = ax[0, 0].contourf(
            x_eval, y_eval, eval_slice, 100,
            vmin=0, vmax=max_ref_dose, cmap=plt.get_cmap('viridis'))
        ax[0, 0].set_title(u"评估剂量 (y = {0:.1f} mm)".format(z_location))
        fig.colorbar(c00, ax=ax[0, 0], label=u'剂量 (cGy)')
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_xlabel('x (mm)')
        ax[0, 0].set_ylabel('z (mm)')

        c01 = ax[0, 1].contourf(
            x_ref, y_ref, ref_slice, 100,
            vmin=0, vmax=max_ref_dose, cmap=plt.get_cmap('viridis'))
        ax[0, 1].set_title(u"参考剂量 (y = {0:.2f} mm)".format(z_location))
        fig.colorbar(c01, ax=ax[0, 1], label=u'剂量 (cGy)')
        ax[0, 1].invert_yaxis()
        ax[0, 1].set_xlabel('x (mm)')
        ax[0, 1].set_ylabel('z (mm)')

        c10 = ax[1, 0].contourf(
            x_ref, y_ref, diff, 100,
            # vmin=-max_diff, vmax=max_diff, cmap=plt.get_cmap('seismic'))
            vmin= 0, vmax= 1, cmap=plt.get_cmap('seismic'))
        ax[1, 0].set_title(u"剂量差")
        fig.colorbar(c10, ax=ax[1, 0], label='[Dose Eval] - [Dose Ref] (cGy)')
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_xlabel('x (mm)')
        ax[1, 0].set_ylabel('z (mm)')

        c11 = ax[1, 1].contourf(
            x_ref, y_ref, gamma_slice, 100,
            vmin=0, vmax=2, cmap=plt.get_cmap('coolwarm'))
        ax[1, 1].set_title("Local Gamma (3 % / 2mm)")
        fig.colorbar(c11, ax=ax[1, 1], label='Gamma')
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_xlabel('x (mm)')
        ax[1, 1].set_ylabel('z (mm)')

        plotName = patientID + 'gamma_{}.png'.format(i)
        plt.savefig(os.path.join(data_path, plotName), dpi=300)
        plt.close()
        # plt.show()
        # print("\n")

        #plot x-y axie profile
        ax_y = plt.subplot2grid((3,3),(0,0),rowspan=2)
        x_loc = round(len(x_ref) / 2)
        ref_y_line = ref_slice[:, x_loc]
        eva_y_line = eval_slice[:, x_loc]
        ax_y.plot(ref_y_line,y_ref, 'r+',  eva_y_line,y_ref, 'y')
        ax_y.invert_xaxis()
        ax_y.set_xlabel(u'剂量(cGy)')
        ax_y.set_ylabel(u'位置(mm)')
        # ax_y.xticks(y_ref,rotation='vertical')


        ax_xy = plt.subplot2grid((3,3),(0,1),colspan=2,rowspan=2)
        ax_xy.contourf(
            x_ref, y_ref, ref_slice, 100,
            vmin=0, vmax=max_ref_dose, cmap=plt.get_cmap('viridis'))
        ax_xy.set_title(u"剂量 (y = {0:.2f} mm)".format(z_location))
        ax_xy.axhline(y_ref[round(len(y_ref)/2)])
        ax_xy.axvline(x_ref[round(len(x_ref)/2)])
        #fig.colorbar(c01, ax=ax[0, 1], label=u'剂量 (cGy)')
        ax_xy.invert_yaxis()
        # ax_xy.set_xlabel('x (mm)')
        # ax_xy.set_ylabel('z (mm)')

        ax_x = plt.subplot2grid((3,3),(2,1),colspan=2)
        y_loc = round(len(y_ref) / 2)
        ref_x_line = ref_slice[y_loc, :]
        eva_x_line = eval_slice[y_loc, :]
        # ax_y.plot(y_ref, ref_x_line, 'r', y_ref, eva_x_line, 'b')
        ax_x.plot(x_ref,ref_x_line, 'r+',  x_ref, eva_x_line, 'y')
        # x_loc = x_ref[round(len(x_ref)/2)]
        # ref_y_line = ref_slice[:,round(len(x_ref)/2)]
        # eva_y_line = eval_slice[:,round(len(x_ref)/2)]
        # ax_x.plot(y_ref,ref_y_line,'r+',y_ref,eva_y_line,'y')
        ax_x.invert_yaxis()
        ax_x.set_ylabel(u'剂量(cGy)')
        ax_x.set_xlabel(u'位置(mm)')
        # ax_x.legend()

        # plt.show()
        plotName = patientID + 'xyprofile_{}.png'.format(i)
        plt.savefig(os.path.join(data_path, plotName), dpi=300)
        plt.close()

def calcDiff(refRS,refRD,evaRS,evaRD):
    structRS = evaRS.GetStructures()
    for (key, Roi) in structs.items():
        print('============================')
        print(key, Roi['name'])
        if Roi['type'] == 'MARKER' or 'Patient' in Roi['name'] or 'Opt.nerve' in Roi['name']:
            continue
        if 'Len' in Roi['name'] or 'plan' in Roi['name'] or '1+2' in Roi['name'] or 'NT' == Roi['name']:
            continue
        elif Roi['type'] == 'ORGAN':
            logging.info('getdvH')
            dvh_tps = getTPSDVH(tpsDVHsDir, patientInfo.MedicalRecordNumber, Roi['name'])
            if dvh_tps:
                # dvh_cal = dvhcalc.get_dvh(Rs.ds, Rd.ds, key)
                dvh_cal = dvhcalc.get_dvh(Rs.ds, Rd.ds, key, interpolation_resolution=(4 / 4),
                                          interpolation_segments_between_planes=2, use_structure_extents=True)

            if dvh_tps and dvh_cal:
                logging.info('abs')
                logging.info(dvh_cal.volume)
                logging.info(dvh_tps.volume)

                dvhdata_cal = dvhdata(dvh_cal)
                dvhdata_tps = dvhdata(dvh_tps)
                dvhdata_cal.cal_nrmsd(dvhdata_tps, resultData)
                dvhdata_cal.getDifferences(dvhdata_tps, resultData)
                dvhdata_cal = None
                dvhdata_tps = None
            dvh_tps = None
            dvh_cal = None

def calcCompTPS(tpsData_path,output_path):
    refRS = ''
    refRD = ''
    evaRS = ''
    evaRD = ''

    refRSList = glob(os.path.join(tpsData_path,'RS*.dcm'))
    if refRSList:
        refRS = refRSList[0]
    refRDList = glob(os.path.join(tpsData_path,'RD*.dcm'))
    if refRDList:
        refRD = refRSList[0]
    evaRSList = glob(os.path.join(output_path, 'RS*.dcm'))
    if evaRSList:
        evaRS = evaRSList[0]
    evaRDList = glob(os.path.join(output_path, 'RD*.dcm'))
    if evaRDList:
        evaRD = evaRDList[0]
    if refRD and evaRD:
        calcGamma(refRD, evaRD)
    if refRS and refRD and evaRS and evaRD:
        calcDiff(refRS,refRD,evaRS,evaRD)

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
        tpsDVHsDir = os.path.join(workingPath, 'Accuracy', 'tps_dcm_CIRS/')
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
        CleaningPatientPool(os.path.join(inputfolder,patient))
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
            else:
                print('plan analysed! skip\n')
            tpsData_path = os.path.join(tpsDVHsDir,patientInfo.MedicalRecordNumber)
            if os.path.exists(output_path) and os.path.exists(tpsData_path):
                calcCompTPS(tpsData_path,output_path)


