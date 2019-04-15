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


class buildPatientPlan(object):
    def __init__(self):
        pass

    def buildPlan(self, planRawDataDict):
        planRawData = planRawDataDict.patientBaseDict
        if 'planListRawData' in planRawData:
            for plan in planRawData.planListRawData:
                logging.info('planName:%s', plan.PlanName)
                refImageID = plan.PrimaryCTImageSetID
                refImageHder = None
                refImageData = None
                for image in planRawData.imageSetListRawData:
                    if image.ImageSetID == refImageID:
                        refImageHder = image.CTHeader
                        refImageData = image.CTData
                        break

                setupPosition = None
                roiShiftVector = None
                if refImageHder:
                    setupPosition = refImageHder.patient_position
                    roiShiftVector = self.getStructShift(refImageHder)

                if 'planPointsRawData' in plan.planData:
                    pointsDict = self.getPoints(
                        plan.planData.planPointsRawData, setupPosition, roiShiftVector)

                if 'planROIsRawData' in plan.planData:
                    contourDict = self.getContours(
                        plan.planData.planROIsRawData, setupPosition, roiShiftVector)

                if 'planTrialsRawData' in plan.planData:
                    trialData = self.getTrialData(
                        plan.planData.planTrialsRawData)

    def getTrialData(self, planTrialData):
        if 'trialListRawData' in planTrialData:
            currentTrailList = planTrialData['TrialList']
            logging.info("PlanHave %d Trials", len(currentTrailList))
            for currentTrail in currentTrailList:
                logging.info("Trial:%s", currentTrail.Name)
                data = self.readTrialMaxtrixData(
                    planDirAbsPath, currentTrail, planDict)
        else:
            logging.info('======================')
            logging.info("Trial:%s", planTrialData.Trial.Name)
            data = self.readTrialMaxtrixData(
                planDirAbsPath, planTrialData['Trial'], planDict)

    def getContours(self, planControurData, patient_position, shiftVector):
        roiListData =
        if 'roiList' in planControurData:
            for curROI in planControurData['roiList']:
                logging.info("ROIName:%s", curROI.name)
                logging.info("ROICTName:%s", curROI.volume_name)
                logging.info('num_curve:%d', len(curROI.num_curve))
                for i, curve in enumerate(curROI.num_curve, 0):
                    curvePoints = []
                    for curr_points in curve.Points:
                        if patient_position == 'HFS':
                            #curr_points = [str(float(curr_points[0])*10), str(float(curr_points[1])*10), str(float(curr_points[2])*10)]
                            curr_points = [str(float(curr_points[0]) * 10 - shiftVector[0]), str(-float(
                                curr_points[1]) * 10 - shiftVector[1]), str(-float(curr_points[2]) * 10)]
                        elif patient_position == 'HFP':
                            curr_points = [str(-float(curr_points[0]) * 10 - shiftVector[0]), str(
                                float(curr_points[1]) * 10 - shiftVector[1]), str(-float(curr_points[2]) * 10)]
                        elif patient_position == 'FFP':
                            curr_points = [str(float(curr_points[0]) * 10 - shiftVector[0]), str(
                                float(curr_points[1]) * 10 - shiftVector[1]), str(float(curr_points[2]) * 10)]
                        elif patient_position == 'FFS':
                            curr_points = [str(-float(curr_points[0]) * 10 - shiftVector[0]), str(-float(
                                curr_points[1]) * 10 - shiftVector[1]), str(float(curr_points[2]) * 10)]
                        curvePoints = curvePoints + curr_points

    def getPoints(self, planPointsData, patient_position, shiftVector):
        pointData = BoxList()
        for point in planPointsData.PoiList:
            x_coord = 0.0
            y_coord = 0.0
            z_coord = 0.0
            if patient_position == 'HFS':
                x_coord = str(float(point.XCoord) * 10)
                y_coord = str(-float(point.YCoord) * 10)
                z_coord = str(-float(point.ZCoord) * 10)
            elif patient_position == 'HFP':
                x_coord = str(-float(point.XCoord) * 10)
                y_coord = str(float(point.YCoord) * 10)
                z_coord = str(-float(point.ZCoord) * 10)
            elif patient_position == 'FFS':
                x_coord = str(-float(point.XCoord) * 10)
                y_coord = str(-float(point.YCoord) * 10)
                z_coord = str(float(point.ZCoord) * 10)
            elif patient_position == 'FFP':
                x_coord = str(float(point.XCoord) * 10)
                y_coord = str(-float(point.YCoord) * 10)
                z_coord = str(float(point.ZCoord) * 10)
            point.xCoord = x_coord
            point.yCoord = y_coord
            point.zCoord = z_coord
            pointData.append(point)
            printformatter = '\"%s:' + \
                (point.CoordinateFormat + ",") * 3 + '\"'
            logging.info(printformatter, point.Name, float(
                x_coord), float(y_coord), float(z_coord))
        return pointData
    ####################################################################################################################################################
    # Function: getstructshift()
    # Purpose: reads in values from ImageSet_0.header to get x and y shift
    ####################################################################################################################################################

    def getStructShift(self, imgHdr):
        xshift = 0
        yshift = 0
        zshift = 0

        # imgHdr = pinn2Json().read(imageHeadFile)
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
