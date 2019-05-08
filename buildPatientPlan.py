#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import time

import imView
import numpy as np
from box import BoxList
from pinn2Json import pinn2Json
from parsePatientPlan import parsePatientPlan
import pydicom
from pydicom import Dataset, Sequence, FileDataset
from dicompylercore import dicomparser, dvhcalc
from dicompylercore.dvh import DVH
from shapely.geometry import Polygon


class buildPatientPlan(object):
    def __init__(self):
        pass

    def createStructDS(self, planData, ImageInfoUIDs, setupPosition, roiShiftVector):
        print("Creating Data structure")
        # create RS SOPInstanceUID
        structSOPInstanceUID = pydicom.uid.generate_uid()
        structSeriesInstanceUID = pydicom.uid.generate_uid()

        # get image header info from ImageSet_0.ImageInfo
        structFrameUID = ''
        structStudyInstanceUID = ''
        structSeriesUID = ''
        structClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        if "ImageInfoList" in ImageInfoUIDs:
            structFrameUID = ImageInfoUIDs.ImageInfoList[0].FrameUID
            structStudyInstanceUID = ImageInfoUIDs.ImageInfoList[0].StudyInstanceUID
            structSeriesUID = ImageInfoUIDs.ImageInfoList[0].SeriesUID
            # structClassUID = ImageInfoUIDs.ImageInfoList[0].ClassUID

        # Populate required values for file meta information
        file_meta = Dataset()
        # RT Structure Set Storage
        file_meta.MediaStorageSOPClassUID = structClassUID
        file_meta.MediaStorageSOPInstanceUID = structSOPInstanceUID
        structfilename = "RS." + structSOPInstanceUID + ".dcm"
        # this value remains static since implementation for creating file is the same
        file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.75006884747854523615841001'
        # Create the FileDataset instance (initially no data elements, but file_meta supplied)
        ds = FileDataset(structfilename, {}, file_meta=file_meta,
                         preamble=b'\x00' * 128)
        # print(file_meta.preamble)

        # add info_data,basic patientinfo
        # [0008,0005] - [0008,0018]
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.InstanceCreationDate = time.strftime("%Y%m%d")
        ds.InstanceCreationTime = time.strftime("%H%M%S")
        ds.SOPClassUID = structClassUID
        ds.SOPInstanceUID = structSOPInstanceUID
        ds.Modality = 'RTSTRUCT'
        ds.AccessionNumber = ""
        ds.Manufacturer = 'Pinnalce3'  # from sample dicom file, maybe should change?
        # not sure where to get information for this element can find this and read in from
        ds.StationName = "adacp3u7"
        # ds.ManufacturersModelName = 'Pinnacle3'
        ds = self.modifyPatientInfo(ds, planData)

        # [0008,1110]
        ds.ReferencedStudySequence = Sequence()
        ReferencedStudy1 = Dataset()
        ds.ReferencedStudySequence.append(ReferencedStudy1)
        # Study Component Management SOP Class (chosen from template)
        ds.ReferencedStudySequence[0].ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
        ds.ReferencedStudySequence[0].ReferencedSOPInstanceUID = structStudyInstanceUID
        # ds.StudyInstanceUID = StudyInstanceUID
        print("Setting structure file study instance: " +
              str(structStudyInstanceUID))
        # [0020,000d]
        ds.StudyInstanceUID = structStudyInstanceUID
        # [0020,000e]
        ds.SeriesInstanceUID = structSeriesInstanceUID

        # [3006,0010]
        ds.ReferencedFrameOfReferenceSequence = Sequence()
        ReferencedFrameofReference1 = Dataset()
        ds.ReferencedFrameOfReferenceSequence.append(
            ReferencedFrameofReference1)
        ds.ReferencedFrameOfReferenceSequence[0].FrameofReferenceUID = structFrameUID
        # [3006,0012]
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence = Sequence(
        )
        RTReferencedStudy1 = Dataset()
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence.append(
            RTReferencedStudy1)
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
            0].ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
            0].ReferencedSOPInstanceUID = structStudyInstanceUID
        # ds.StudyInstanceUID = StudyInstanceUID
        # [3006,0014]
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence = Sequence(
        )
        RTReferencedSeries1 = Dataset()
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence.append(
            RTReferencedSeries1)
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
            0].RTReferencedSeriesSequence[0].SeriesInstanceUID = structSeriesUID

        # [3006,0016]
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
            0].RTReferencedSeriesSequence[0].ContourImageSequence = Sequence()

        # [fffe,e000]
        for i, value in enumerate(ImageInfoUIDs.ImageInfoList, 1):
            exec("ContourImage%d = Dataset()" % i)
            exec(
                "ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(ContourImage%d)" % i)
            ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[i - 1].ReferencedSOPClassUID = value.ClassUID
            ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[i - 1].ReferencedSOPInstanceUID = value.InstanceUID
            # exec("del ContourImage%d" % i)

        # [3006,0020]
        roiListData = planData.planROIsRawData.roiList
        ds.StructureSetROISequence = Sequence()
        for i, value in enumerate(roiListData, 1):
            exec("ROISet%d = Dataset()" % i)
            exec("ds.StructureSetROISequence.append(ROISet%d)" % i)
            ds.StructureSetROISequence[i - 1].ROIName = value.name
            ds.StructureSetROISequence[i - 1].ROINumber = i
            ds.StructureSetROISequence[i -
                                       1].ReferencedFrameOfReferenceUID = structFrameUID
            if 'volume' in value:
                ds.StructureSetROISequence[i - 1].ROIVolume = value.volume
            ds.StructureSetROISequence[i -
                                       1].ROIGenerationAlgorithm = value.roiinterpretedtype

        # [3006,0039]get each ROI
        ds.ROIContourSequence = Sequence()
        for i, value in enumerate(roiListData, 1):
            exec("ContourSequence%d = Dataset()" % i)
            exec("ds.ROIContourSequence.append(ContourSequence%d)" % i)
            ds.ROIContourSequence[i - 1].ROIDisplayColor = [0, 255, 0]
            ds.ROIContourSequence[i - 1].ReferencedROINumber = i

            # get all curves in current ROI
            ds.ROIContourSequence[i - 1].ContourSequence = Sequence()
            planROIsCurvesList = value.num_curve
            # get each ROI_Curvers
            for j, data in enumerate(planROIsCurvesList, 1):
                exec("CurvesPoint%d = Dataset()" % j)
                exec(
                    "ds.ROIContourSequence[i - 1].ContourSequence.append(CurvesPoint%d)" % j)
                # [3006,0040]
                ds.ROIContourSequence[i - 1].ContourSequence[j -
                                                             1].ContourImageSequence = Sequence()
                coutourImage1 = Dataset()
                ds.ROIContourSequence[i - 1].ContourSequence[j -
                                                             1].ContourImageSequence.append(coutourImage1)
                ds.ROIContourSequence[i - 1].ContourSequence[j -
                                                             1].ContourImageSequence[0].ReferencedSOPClassUID = structClassUID
                ds.ROIContourSequence[i - 1].ContourSequence[j - 1].ContourImageSequence[0].ReferencedSOPInstanceUID = self.getCTInstanceUID(
                    data.Points[0], setupPosition, ImageInfoUIDs)

                # [3006,0042]
                ds.ROIContourSequence[i - 1].ContourSequence[j -
                                                             1].ContourGeometricType = "CLOSED_PLANAR"
                ds.ROIContourSequence[i - 1].ContourSequence[j -
                                                             1].NumberOfContourPoints = data.num_points
                # get each ROI_Curves_Points, using data.Points
                ds.ROIContourSequence[i - 1].ContourSequence[j - 1].ContourData = self.getContourCurvePoints(
                    data.Points, setupPosition, roiShiftVector)

        # [3006,0080]
        ds.RTROIObservationsSequence = Sequence()
        for i, current_roi in enumerate(ds.StructureSetROISequence, 1):
            exec("Observation%d = Dataset()" % i)
            exec("ds.RTROIObservationsSequence.append(Observation%d)" % i)
            ds.RTROIObservationsSequence[i -
                                         1].ObservationNumber = current_roi.ROINumber
            ds.RTROIObservationsSequence[i -
                                         1].ReferencedROINumber = current_roi.ROINumber
            ds.RTROIObservationsSequence[i - 1].RTROIInterpretedType = 'ORGAN'
            ds.RTROIObservationsSequence[i - 1].ROIInterpreter = ""

        # find out where to get if its been approved or not
        ds.ApprovalStatus = 'UNAPPROVED'
        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # # Create the FileDataset instance (initially no data elements, but file_meta supplied)
        # structds = FileDataset(structfilename, {},
        #                  file_meta=ds, preamble=b'\x00' * 128)

        # structfilepath=outputfolder + patientfolder + "/" + structfilename
        # structds.save_as("structfilepath")
        # print("Structure file being saved\n")
        ds.save_as(os.getenv('HOME') + '/PinnWork/' + structfilename)
        #
        # dcmds = pydicom.dcmread(ds)
        # print(dcmds)
        return ds

    def modifyPatientInfo(self, ds, rawData):
        ds.StructureSetName = 'POIandROI'
        ds.SeriesNumber = '1'
        ds.PatientsName = rawData.FirstName + rawData.LastName + rawData.MiddleName
        ds.PatientID = rawData.MedicalRecordNumber

        ds.ReferringPhysiciansName = rawData.Physician
        ds.PhysiciansOfRecord = rawData.Physician
        ds.PatientsSex = rawData.Gender
        # ds.SoftwareVersions = rawData.PinnacleVersionDescription
        ds.ManufacturersModelName = 'Pinnacle3'

        # birthday
        dobstr = rawData.DateOfBirth
        if dobstr:
            if '-' in dobstr[0]:
                dob_list = dobstr[0].split('-')
            elif '/' in dobstr[0]:
                dob_list = dobstr[0].split('/')
            else:
                dob_list = dobstr[0].split(' ')
            dob = ""
            for num in dob_list:
                if len(num) == 1:
                    num = '0' + num
                if num == dob_list[-1] and len(num) == 2:
                    num = "19" + num
                dob = dob + num
            ds.PatientsBirthDate = dob
        # date and time
        dateAndTime = rawData.TimeStamp
        arr = dateAndTime.split()
        ds.StructureSetDate = arr[1] + arr[2] + arr[-1]
        ds.StructureSetTime = arr[-2].replace(':', '')
        ds.StudyDate = ds.StructureSetDate
        ds.StudyTime = ds.StructureSetTime

        return ds

    def createImagesDS(self, ds, ImageInfoUIDs):
        # imageUIDs = pinn2Json().read(imageInfoFile)
        if "ImageInfoList" in ImageInfoUIDs:
            ds.FrameUID = ImageInfoUIDs.ImageInfoList[0].FrameUID
            ds.StudyInstanceUID = ImageInfoUIDs.ImageInfoList[0].StudyInstanceUID
            ds.SeriesUID = ImageInfoUIDs.ImageInfoList[0].SeriesUID
            ds.ClassUIDD = ImageInfoUIDs.ImageInfoList[0].ClassUID

    def buildPlan(self, planRawDataDict):
        planRawData = planRawDataDict.patientBaseDict
        if 'planListRawData' in planRawData:
            for plan in planRawData.planListRawData:
                logging.info('planName:%s', plan.PlanName)
                refImageID = plan.PrimaryCTImageSetID
                refImageHder = None
                refImageInfo = None
                refImageData = None
                for image in planRawData.imageSetListRawData:
                    if image.ImageSetID == refImageID:
                        refImageHder = image.CTHeader
                        refImageInfo = image.CTInfo
                        refImageData = image.CTData
                        break

                setupPosition = None
                roiShiftVector = None
                if refImageHder:
                    setupPosition = refImageHder.patient_position
                    roiShiftVector = self.getStructShift(refImageHder)

                # if 'planPointsRawData' in plan.planData:
                #     pointsDict = self.getPoints(
                #         plan.planData.planPointsRawData, setupPosition, roiShiftVector)

                if 'planROIsRawData' in plan.planData:
                    # contourDict = self.getContours(
                        # plan.planData.planROIsRawData, setupPosition, roiShiftVector)

                    # createStructDS(patientInfoDict, ImageInfoUIDs, planROIsRawData,setupPosition, roiShiftVector)
                    RS_ds = self.createStructDS(plan.planData,
                                                refImageInfo,
                                                setupPosition,
                                                roiShiftVector)
                    self.getROIsVolumes(RS_ds)
                    logging.info('=========================')
                    tps_ds = pydicom.read_file('/home/peter/PinnWork/Accuracy/DCM_38169/RS.dcm')
                    self.getROIsVolumes(tps_ds)

                print('end')
                # if 'planTrialsRawData' in plan.planData:
                #     trialData = self.getTrialData(
                #         plan.planData.planTrialsRawData)
    def getROIsVolumes(self,RS_ds):
        rsObject = dicomparser.DicomParser(RS_ds)
        structs = rsObject.GetStructures()
        for (key, Roi) in structs.items():
            if Roi['name'] == 'Cord':
                planes = rsObject.GetStructureCoordinates(key)
                thickness = rsObject.CalculatePlaneThickness(planes)
                # volume = rsObject.CalculateStructureVolume(planes,thickness)
                volume = self.CalculateStructureVolume(planes, thickness)
                logging.info('key=%d,name=%s,volume=%s', key, Roi['name'], str(volume))

    def CalculateStructureVolume(self, coords, thickness):
        """Calculates the volume of the given structure.

        Parameters
        ----------
        coords : dict
            Coordinates of each plane of the structure
        thickness : float
            Thickness of the structure
        """

        # if not shapely_available:
        #     print("Shapely library not available." +
        #           " Please install to calculate.")
        #     return 0

        class Within(Polygon):
            def __init__(self, o):
                self.o = o

            def __lt__(self, other):
                return self.o.within(other.o)

        s = 0
        for i, z in enumerate(sorted(coords.items())):
            # Skip contour data if it is not CLOSED_PLANAR
            logging.info(z[0])
            if z[1][0]['type'] != 'CLOSED_PLANAR':
                continue
            polygons = []
            contours = [[x[0:2] for x in c['data']] for c in z[1]]
            for contour in contours:
                p = Polygon(contour)
                polygons.append(p)
            # Sort polygons according whether they are contained
            # by the previous polygon
            if len(polygons) > 1:
                ordered_polygons = sorted(polygons, key=Within, reverse=True)
            else:
                ordered_polygons = polygons
            for ip, p in enumerate(ordered_polygons):
                pa = 0
                if ((i == 0) or (i == len(coords.items()) - 1)) and \
                        not (len(coords.items()) == 1):
                    pa += (p.area // 2)
                else:
                    pa += p.area
                # Subtract the volume if polygon is contained within the parent
                # and is not the parent itself
                if p.within(ordered_polygons[0]) and \
                        (p != ordered_polygons[0]):
                    s -= pa
                else:
                    s += pa
                logging.info(pa)
        vol = s * thickness / 1000
        logging.info(thickness)
        logging.info('Total_volume')
        logging.info(vol)
        return vol

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
        # roiListData =
        if 'roiList' in planControurData:
            for curROI in planControurData['roiList']:
                logging.info("ROIName:%s", curROI.name)
                logging.info("ROICTName:%s", curROI.volume_name)
                logging.info('num_curve:%d', len(curROI.num_curve))
                for i, curve in enumerate(curROI.num_curve, 0):
                    curvePoints = []
                    for curr_points in curve.Points:
                        if patient_position == 'HFS':
                            # curr_points = [str(float(curr_points[0])*10), str(float(curr_points[1])*10), str(float(curr_points[2])*10)]
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

    def getContourCurvePoints(self, planPointsData, patient_position, shiftVector):
        Points = []
        for point in planPointsData:
            x_coord = 0.0
            y_coord = 0.0
            z_coord = 0.0
            if patient_position == 'HFS':
                x_coord = str(float(point[0]) * 10)
                y_coord = str(-float(point[1]) * 10)
                z_coord = str(-float(point[2]) * 10)
            elif patient_position == 'HFP':
                x_coord = str(-float(point[0]) * 10)
                y_coord = str(float(point[1]) * 10)
                z_coord = str(-float(point[2]) * 10)
            elif patient_position == 'FFS':
                x_coord = str(-float(point[0]) * 10)
                y_coord = str(-float(point[1]) * 10)
                z_coord = str(float(point[2]) * 10)
            elif patient_position == 'FFP':
                x_coord = str(float(point[0]) * 10)
                y_coord = str(-float(point[1]) * 10)
                z_coord = str(float(point[2]) * 10)
            Points.append(x_coord)
            Points.append(y_coord)
            Points.append(z_coord)
            # logging.info(x_coord, y_coord, z_coord)
        return Points

    def getCTInstanceUID(self, point_data, patient_position, ImageInfoUIDs):
        # CT location th<0.02cm
        Z_throth = 0.02
        # units are CM, not mm
        point_ZCoord = point_data[-1]

        for ct in ImageInfoUIDs.ImageInfoList:
            if patient_position == 'HFS' or patient_position == 'HFP':
                CT_Zcoord = -float(ct.CouchPos)
            elif patient_position == 'FFS' or patient_position == 'FFP':
                CT_Zcoord = float(ct.CouchPos)
            if abs(point_ZCoord - CT_Zcoord) < Z_throth:
                return ct.InstanceUID

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
    # inputfolder = os.path.join(workingPath, 'DCM_Pinn')
    #
    # planData = parsePatientPlan(os.path.join(inputfolder, 'Patient_35995'))
    planData = parsePatientPlan(os.path.join(workingPath, 'Accuracy/Mount_496285/','Patient_38169'))
    planData.getPatientDict()
    planData.printPatientDict()

    newPlanObject = buildPatientPlan()
    newPlanObject.buildPlan(planData)
