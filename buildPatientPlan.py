#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

import imView
import numpy as np
from box import BoxList
from pinn2Json import pinn2Json
from parsePatientPlan import parsePatientPlan

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


class buildPatientPlan(object):
    def __init__(self):
        pass

    def buildPlan(self, planRawDataDict):
        for plan in planRawDataDict['PlanList']:
            Get

    def getContours(self, planControurData, patient_position, shiftVoctor):
        if 'roiList' in planControurData:
            roiList = planControurData['roiList']
            for curROI in roiList:
                logging.info(curROI.name)
                logging.info(curROI.num_curve)

    ####################################################################################################################################################
    # Function: getstructshift()
    # Purpose: reads in values from ImageSet_0.header to get x and y shift
    ####################################################################################################################################################
    def getStructShift(self, imageHeadFile):
        xshift = 0
        yshift = 0
        zshift = 0

        imgHdr = pinn2Json().read(imageHeadFile)
        x_dim = float(imgHdr.x_dim)
        y_dim = float(imgHdr.y_dim)
        z_dim = float(imgHdr.z_dim)
        xpixdim = float(imgHdr.x_pixdim)
        ypixdim = float(imgHdr.y_pixdim)
        zpixdim = float(imgHdr.z_pixdim)

        # pinnacle version differences
        # xstart = float(imgHdr.x_start_dicom)
        # ystart = float(imgHdr.y_start_dicom)

        xstart = float(imgHdr.x_start)
        ystart = float(imgHdr.y_start)
        zstart = float(imgHdr.z_start)
        patient_position = imgHdr.patient_position
        # with open("%s%s/ImageSet_%s.header" % (Inputf, patientfolder, imagesetnumber), "rt", encoding=u'utf-8',
        #           errors='ignore') as f2:
        #     for line in f2:
        #         if "x_dim =" in line:
        #             x_dim = float((line.split(" ")[-1]).replace(';', ''))
        #         if "y_dim =" in line:
        #             y_dim = float((line.split(" ")[-1]).replace(';', ''))
        #         if "x_pixdim =" in line:
        #             xpixdim = float((line.split(" ")[-1]).replace(';', ''))
        #         if "y_pixdim =" in line:
        #             ypixdim = float((line.split(" ")[-1]).replace(';', ''))
        #         if "x_start =" in line and "index" not in line:
        #             xstart = float((line.split(" ")[-1]).replace(';', ''))
        #             print("xstart = ", xstart)
        #         if "y_start =" in line:
        #             ystart = float((line.split(" ")[-1]).replace(';', ''))
        #         if "z_dim =" in line:
        #             z_dim = float((line.split(" ")[-1]).replace(';', ''))
        #         if "z_pixdim =" in line:
        #             zpixdim = float((line.split(" ")[-1]).replace(';', ''))
        #         if "z_start =" in line and "index" not in line:
        #             zstart = float((line.split(" ")[-1]).replace(';', ''))
        if patient_position == 'HFS':
            xshift = ((x_dim * xpixdim / 2) + xstart) * 10
            yshift = -((y_dim * ypixdim / 2) + ystart) * 10
            zshift = -((z_dim * zpixdim / 2) + zstart) * 10
        elif patient_position == 'HFP':
            xshift = -((x_dim * xpixdim / 2) + xstart) * 10
            yshift = ((y_dim * ypixdim / 2) + ystart) * 10
            zshift = -((z_dim * zpixdim / 2) + zstart) * 10
        elif patient_position == 'FFP':
            xshift = ((x_dim * xpixdim / 2) + xstart) * 10
            yshift = ((y_dim * ypixdim / 2) + ystart) * 10
            zshift = ((z_dim * zpixdim / 2) + zstart) * 10
        elif patient_position == 'FFS':
            xshift = -((x_dim * xpixdim / 2) + xstart) * 10
            yshift = -((y_dim * ypixdim / 2) + ystart) * 10
            zshift = ((z_dim * zpixdim / 2) + zstart) * 10

        logging.info("X shift = %s", xshift)
        logging.info("Y shift = %s", yshift)
        logging.info("Z shift = %s", zshift)
        return [xshift, yshift, zshift]

    def getTrialData(self):
        if os.path.isfile(os.path.join(planDirAbsPath, 'plan.Trial')):
            logging.info('======================')
            logging.info('Reading Trials, waiting..... ')
            planTrialData = pinn2Json().read(
                os.path.join(planDirAbsPath, 'plan.Trial'))
            # if 'TrialList' in planTrialData:
            #     currentTrailList = planTrialData['TrialList']
            #     logging.info("PlanHave %d Trials", len(currentTrailList))
            #     for currentTrail in currentTrailList:
            #         logging.info("Trial:%s", currentTrail.Name)
            #         # #data = self.readTrialMaxtrixData(
            #         #     planDirAbsPath, currentTrail, planDict)
            # else:
            #     logging.info('======================')
            #     logging.info("Trial:%s", planTrialData.Trial.Name)
            #     data = self.readTrialMaxtrixData(
            #         planDirAbsPath, planTrialData['Trial'], planDict)

    def readTrialMaxtrixData(self, trialBasePath, curTrial, planDict):

        planPoints = planDict['Points']
        doseHdr = curTrial.DoseGrid
        dose = np.zeros((doseHdr.Dimension.Z,
                         doseHdr.Dimension.Y,
                         doseHdr.Dimension.X))
        for pInd, ps in enumerate(curTrial.PrescriptionList):
            logging.info('%s:%d:%d', ps.Name, ps.PrescriptionDose,
                         ps.NumberOfFractions)

            for bInd, bm in enumerate(curTrial.BeamList):
                try:
                    # Get the name of the file where the beam dose is saved -
                    # PREVIOUSLY USED DoseVarVolume ?
                    doseFile = os.path.join(trialBasePath,
                                            "plan.Trial.binary.%03d" %
                                            int(bm.DoseVolume.split('-')[1]))

                    # Read the dose from the file
                    bmDose = np.fromfile(doseFile, dtype='float32')

                    if bmDose.nbytes == 0:
                        raise DoseInvalidException('')

                except IOError or SystemError:
                    raise DoseInvalidException('')
                # Reshape to a 3D array
                bmDose = bmDose.reshape((doseHdr.Dimension.Z,
                                         doseHdr.Dimension.Y,
                                         doseHdr.Dimension.X))

                # Solaris uses big endian schema.
                # Almost everything else is little endian
                if sys.byteorder == 'little':
                    bmDose = bmDose.byteswap(True)

                bmFactor = bm.MonitorUnitInfo.NormalizedDose * \
                    bm.MonitorUnitInfo.CollimatorOutputFactor * \
                    bm.MonitorUnitInfo.TotalTransmissionFraction
                dosePerMU = 0.665
                # getting dose/Mu from the plan.Pinnacle.Machines file
                # dosePerMU = self.getDosePerMU()
                MUs = bm.MonitorUnitInfo.PrescriptionDose / \
                    (bmFactor * dosePerMU)
                logging.info('%s:%d', bm.Name, MUs)

                # Weight the dose cube by the beam weight
                dose += (bmDose * bm.Weight)

                # rescale dose to prescriptionDose
                totalPrescriptionDose = ps.PrescriptionDose * ps.NumberOfFractions
                doseAtPoint = totalPrescriptionDose * 1
                if ps.Name == bm.PrescriptionName:
                    if ps.WeightsProportionalTo == 'Point Dose':
                        for pt in planPoints['PoiList']:
                            if pt.Name == ps.PrescriptionPoint:
                                doseAtPoint = self.doseAtCoord(
                                    dose, doseHdr, pt.XCoord, pt.YCoord, pt.ZCoord)

                                logging.info(doseAtPoint)

                dose = dose * (doseAtPoint / totalPrescriptionDose)
        return dose, doseHdr

    def readDoses(self, planTrialData, planBasePath):
        """
        input:  Read a dose cube for a trial in a given plan and
        return: a numpy array

        Currently tested for:
                (1) Dose is prescribed to a norm point;
                        beam weights are proportional to point dose
                        and control point dose is not stored.
                (2) Dose is prescribed to mean dose of target;
        """
        # trialFile = os.path.join(self.sourceDir, 'plan.Trial')
        # if not os.path.isfile(trialFile):
        #     self.logging.info("not such file %s", trialFile)
        #     return None
        # trialData = pinn2Json().read(trialFile)
        # pts = pinn2Json().read(os.path.join(self.sourceDir, 'plan.Points'))
        if not planTrialData:
            raise IOError

        pts = pinn2Json().read(
            os.path.join(planBasePath, 'plan.Points'))
        # pts = pointsList['']

        trialList = []
        doseDataDict = {}
        dose = None
        if 'TrialList' in planTrialData:
            logging.info(
                ('plan has %d Trials', len(planTrialData.TrialList)))
            for curTrial in planTrialData.TrialList:
                trialList.append(curTrial)
        else:
            logging.info(('plan has %d Trials', len(planTrialData.Trial)))
            trialList.append(planTrialData.Trial)

        for curTrial in trialList:
            doseHdr = curTrial.DoseGrid
            doseData = np.zeros((doseHdr.Dimension.Z,
                                 doseHdr.Dimension.Y,
                                 doseHdr.Dimension.X))

            for bInd, bm in enumerate(curTrial.BeamList):
                try:
                    # Get the name of the file where the beam dose is saved -
                    # PREVIOUSLY USED DoseVarVolume ?
                    doseFile = os.path.join(planBasePath,
                                            "plan.Trial.binary.%03d" %
                                            int(bm.DoseVolume.split('-')[1]))

                    # Read the dose from the file
                    bmDose = np.fromfile(doseFile, dtype='float32')

                    if bmDose.nbytes == 0:
                        raise DoseInvalidException('')

                except IOError or SystemError:
                    raise DoseInvalidException('')
                # Reshape to a 3D array
                bmDose = bmDose.reshape((doseHdr.Dimension.Z,
                                         doseHdr.Dimension.Y,
                                         doseHdr.Dimension.X))

                # Solaris uses big endian schema.
                # Almost everything else is little endian
                if sys.byteorder == 'little':
                    bmDose = bmDose.byteswap(True)

                doseFactor = 1.0

                # Weight the dose cube by the beam weight
                # Assume dose is prescribed to a norm point and beam weights are proportional to point dose
                doseAtPoint = 0.0

                prescriptionPoint = []
                prescriptionDose = []
                prescriptionPointDose = []
                prescriptionPointDoseFactor = []

                for pp in curTrial.PrescriptionList:
                    if pp.Name == bm.PrescriptionName:
                        prescriptionDose.append(
                            pp.PrescriptionDose * pp.NumberOfFractions)
                        if pp.WeightsProportionalTo == 'Point Dose':
                            for pt in pts.PoiList:
                                if pt.Name == pp.PrescriptionPoint:
                                    doseAtPoint = self.doseAtCoord(
                                        bmDose, doseHdr, pt.XCoord, pt.YCoord, pt.ZCoord)
                                    doseFactor = pp.PrescriptionDose * \
                                        pp.NumberOfFractions * \
                                        (bm.Weight * 0.01 / doseAtPoint)

                                    prescriptionPoint.append(
                                        [pt.XCoord, pt.YCoord, pt.ZCoord])
                                    prescriptionPointDose.append(doseAtPoint)
                                    prescriptionPointDoseFactor.append(
                                        doseFactor)
                        elif pp.WeightsProportionalTo == 'ROI Mean':
                            logging.info('get ROI mean dose')

                dose += (bmDose * doseFactor)
        for bm, pD, pp in zip(range(len(prescriptionPointDose)), prescriptionPointDose, prescriptionPoint):
            indPP = coordToIndex(doseHdr, pp[0], pp[1], pp[2])

        return dose, doseHdr
        #         doseData += bmDose
        #     doseDataDict[(curTrial.Name + 'DoseArray')] = doseData
        # return doseDataDict

    def coordToIndex(self, imHdr, xCoord, yCoord, zCoord):
        """
        Convert corrdinate positions to coordinate indices
        """

        # coord in cm from primary image centre
        xCoord -= imHdr.Origin.X
        yCoord = imHdr.Origin.Y + imHdr.Dimension.Y * imHdr.VoxelSize.Y - yCoord
        zCoord -= imHdr.Origin.Z

        # coord now in cm from start of dose cube
        xCoord /= imHdr.VoxelSize.X
        yCoord /= imHdr.VoxelSize.Y
        zCoord /= imHdr.VoxelSize.Z

        # coord now in pixels from start of dose cube
        return xCoord, yCoord, zCoord

    # ----------------------------------------- #

    def doseAtCoord(self, doseData, doseHdr, xCoord, yCoord, zCoord):
        """
        Linearly interpolate the dose at a set of coordinates
        """
        xCoord, yCoord, zCoord = self.coordToIndex(
            doseHdr, xCoord, yCoord, zCoord)

        xP = np.floor(xCoord)
        yP = np.floor(yCoord)
        zP = np.floor(zCoord)

        xF = xCoord - xP
        yF = yCoord - yP
        zF = zCoord - zP

        dose = self.doseAtIndex(doseData, zP, yP, xP) * (1.0 - zF) * (1.0 - yF) * (1.0 - xF) + \
            self.doseAtIndex(doseData, zP, yP, xP + 1) * (1.0 - zF) * (1.0 - yF) * xF + \
            self.doseAtIndex(doseData, zP, yP + 1, xP) * (1.0 - zF) * yF * (1.0 - xF) + \
            self.doseAtIndex(doseData, zP, yP + 1, xP + 1) * (1.0 - zF) * yF * xF + \
            self.doseAtIndex(doseData, zP + 1, yP, xP) * zF * (1.0 - yF) * (1.0 - xF) + \
            self.doseAtIndex(doseData, zP + 1, yP, xP + 1) * zF * (1.0 - yF) * xF + \
            self.doseAtIndex(doseData, zP + 1, yP + 1, xP) * zF * yF * (1.0 - xF) + \
            self.doseAtIndex(doseData, zP + 1, yP + 1, xP + 1) * zF * yF * xF

        return dose

    # ----------------------------------------- #

    def doseAtIndex(self, dose, indZ, indY, indX):
        """
        Return dose at indices.
        Beyond end of dose array return zero
        """
        try:
            dd = dose[indZ, indY, indX]
            if indZ > 0 and indY > 0 and indX > 0:
                return dd
            else:
                return 0.0
        except IndexError:
            return 0.0

    # ----------------------------------------- #


class DoseInvalidException(Exception):
    pass


if __name__ == '__main__':
    workingPath = os.path.join(os.getenv('HOME'), 'PinnWork')
    inputfolder = os.path.join(workingPath, 'Mount_0/')

    planData = parsePatientPlan(os.path.join(inputfolder, 'Patient_28471'))
    planData.getPatientDict()
    planData.printPatientDict()

    newPlanObject = buildPatientPlan()
    newPlanObject.buildPlan(planData)
