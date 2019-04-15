#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

import imView
import numpy as np
from box import BoxList
from pinn2Json import pinn2Json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='convert_whole_patient.log',
                    filemode='w')
# 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class parsePatientPlan(object):
    def __init__(self, sourceDir):
        if not os.path.isdir(sourceDir):
            logging.info('target dir %s not exist!', sourceDir)
            raise 'IOError'

        self.sourceDir = sourceDir
        self.patientBaseDict = BoxList()

    def getPatientDict(self):
        parseDir = self.sourceDir

        if os.path.isfile(os.path.join(parseDir, 'Patient')):
            self.patientBaseDict = pinn2Json().read(os.path.join(parseDir, 'Patient'))
        else:
            logging.error('may be a empty plan! skip')
            return None

        patientImageSetList = BoxList()
        if 'ImageSetList' in self.patientBaseDict:
            # image_set_list = baseDict.get('ImageSetList')
            for imageSet in self.patientBaseDict.ImageSetList:
                logging.info('ImageSet_%s', imageSet.ImageSetID)
                if 'phantom' in imageSet.SeriesDescription:
                    logging.warning('this is Phantom for QA, skip!')
                    continue
                # read CT image set of this plan
                (imageSet['CTHeader'], imageSet['CTData']
                 ) = self.readCT(imageSet.ImageName)
                patientImageSetList.append(imageSet)
        self.patientBaseDict['imageSetListRawData'] = patientImageSetList

        patientPlanList = BoxList()
        if 'PlanList' in self.patientBaseDict:
            # plan_list = baseDict.get('PlanList')
            for plan in self.patientBaseDict.PlanList:
                logging.info('plan_%s,base on ImageSet_%s',
                             plan.PlanID, plan.PrimaryCTImageSetID)
                if 'QA' in plan.PlanName or 'copy' in plan.PlanName:
                    logging.warning('this is Copy or QA plan, skip!')
                else:
                    planDirName = 'Plan_' + (str)(plan.PlanID)
                    logging.info('Reading plan:%s ......', planDirName)
                    plan['planData'] = self.readPlan(planDirName, plan.PrimaryCTImageSetID)
                patientPlanList.append(plan)
        self.patientBaseDict['planListRawData'] = patientPlanList

    def readCT(self, CTName):
        """
        Read a CT cube for a plan
        """
        imHdr = pinn2Json().read(
            os.path.join(self.sourceDir, (CTName + '.header')))

        # Read the data from the file
        imData = np.fromfile(os.path.join(
            self.sourceDir, (CTName + '.img')), dtype='int16')

        # Reshape to a 3D array
        imagePixelLength = imHdr.x_dim * imHdr.y_dim * imHdr.z_dim
        if len(imData) == imagePixelLength:
            imData = imData.reshape((imHdr.z_dim, imHdr.y_dim, imHdr.x_dim))
        else:
            logging.error('img data broke, try dicom types images')
            # parse dicom_dir TODO

        # Solaris uses big endian schema.
        if sys.byteorder == 'little':
            if imHdr.byte_order == 1:
                imData = imData.byteswap(True)
        else:
            if imHdr.byte_order == 0:
                imData = imData.byteswap(True)

        return imHdr, imData

    def readPlan(self, planDirRefPath, planRefImageID):
        """
        read one plan:
            plan.Info
            plan.Points,
            plan.volumeInfo,
            plan.roi,
            plan.Trial,
        :param planDir: plan relative path ./Plan_N
        :return: dict plan
        """

        planDict = None
        planDirAbsPath = os.path.join(self.sourceDir, planDirRefPath)
        if not os.path.isdir(planDirAbsPath):
            logging.error("directory %s not exsit!", planDirAbsPath)
            raise IOError

        if os.path.isfile(os.path.join(planDirAbsPath, 'plan.PlanInfo')):
            logging.info('Reading plan.PlanInfo')
            planDict = pinn2Json().read(
                os.path.join(planDirAbsPath, 'plan.PlanInfo'))
            planDict['planRefImageID'] = planRefImageID
            planRefImageName = 'ImageSet_' + str(planRefImageID)
            planDict['planRefImageName'] = planRefImageName

        if os.path.isfile(os.path.join(planDirAbsPath, 'plan.VolumeInfo')):
            logging.info('Reading plan.VolumeInof')
            planDict['planVolumeInfoRawData'] = pinn2Json().read(
                os.path.join(planDirAbsPath, 'plan.VolumeInfo'))

        if os.path.isfile(os.path.join(planDirAbsPath, 'plan.Points')):
            logging.info('Reading plan.Points')
            planDict['planPointsRawData'] = pinn2Json().read(
                os.path.join(planDirAbsPath, 'plan.Points'))
            # for point in planDict['PointsRawData'].PoiList:
            #     logging.info(point.Name)

        if os.path.isfile(os.path.join(planDirAbsPath, 'plan.roi')):
            logging.info('Reading ROIs, will taking long time, waiting..... ')
            planDict['planROIsRawData'] = pinn2Json().read(os.path.join(planDirAbsPath, 'plan.roi'))
            #self.getContours(planDict['rois'], contourShiftVector)

        # if os.path.isfile(os.path.join(planDirAbsPath, 'plan.Pinnacle.Machines')):
        #     planDict['machines'] = pinn2Json().read(
        #         os.path.join(planDirAbsPath, 'plan.Pinnacle.Machines'))

        if os.path.isfile(os.path.join(planDirAbsPath, 'plan.Trial')):
            logging.info('======================')
            logging.info('Reading Trials, waiting..... ')
            planDict['planTrialsRawData'] = pinn2Json().read(
                os.path.join(planDirAbsPath, 'plan.Trial'))
        return planDict

    def printPatientDict(self):
        """
        Parse the "Patient_XX/Patient" file, get the plan Frame.
        """
        # patientDataDict = self.patientBaseDict
        if self.patientBaseDict:
            logging.info("PatientName:%s%s", self.patientBaseDict.LastName,
                         self.patientBaseDict.FirstName)
            logging.info("MRN:%s", self.patientBaseDict.MedicalRecordNumber)
            logging.info("=========imageList:")
            if 'ImageSetList' in self.patientBaseDict:
                for imageSet in self.patientBaseDict.ImageSetList:
                    logging.info("%s,%s", imageSet.ImageName,
                                 imageSet.ImageSetID)

            logging.info("=======planList:")
            if 'PlanList' in self.patientBaseDict:
                for plan in self.patientBaseDict.PlanList:
                    logging.info("%s,%s,%d", plan.PlanName, plan.PlanID,
                                 plan.PrimaryCTImageSetID)


class DoseInvalidException(Exception):
    pass


if __name__ == '__main__':
    workingPath = os.path.join(os.getenv('HOME'), 'PinnWork')
    inputfolder = os.path.join(workingPath, 'Mount_0/')

    planObject = parsePatientPlan(os.path.join(inputfolder, 'Patient_28471'))
    planObject.getPatientDict()
    planObject.printPatientDict()
