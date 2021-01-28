"""
#add dose file function:
rtDose_path = '.\rtDose'
Img,Mask,Dose = PyrexReader.Img_Bimask(img_path,rtstruct_path, rtDose_path,ROI)
###############################
@author: zhenwei.shi, Maastro##
###############################
Usage:
import PyrexReader

img_path = '.\CTscan'
rtstruct_path = '.\RTstruct'
ROI = Region Of Interest
Img,Mask = PyrexReader.Img_Bimask(img_path,rtstruct_path,ROI)

"""

import pydicom
from dicompylercore import dicomparser, dose
from radiomics import featureextractor
import os
import numpy as np
from skimage import draw
import SimpleITK as sitk
import re
import glob
import six
import matplotlib.path
import logging


# module PyrexReader:


# Match ROI id in RTSTURCT to a given ROI name in the parameter file
def match_ROIid(rtstruct_path, ROI_name):
    mask_vol = Read_RTSTRUCT(rtstruct_path)
    M = mask_vol[0]
    for i in range(len(M.StructureSetROISequence)):
        if str(ROI_name) == M.StructureSetROISequence[i].ROIName:
            ROI_number = M.StructureSetROISequence[i].ROINumber
#            print(ROI_number)
            break
    for ROI_id in range(len(M.StructureSetROISequence)):
        if ROI_number == M.ROIContourSequence[ROI_id].ReferencedROINumber:
            #            print(ROI_number)
            break
    return ROI_id


def ROI_match(ROI, rtstruct_path):  # Literal match ROI
    mask_vol = Read_RTSTRUCT(rtstruct_path)
    M = mask_vol[0]
    target = []
    for i in range(0, len(M.StructureSetROISequence)):
        if re.search(ROI, M.StructureSetROISequence[i].ROIName):
            target.append(M.StructureSetROISequence[i].ROIName)
    if len(target) == 0:
        for j in range(0, len(M.StructureSetROISequence)):
            print(M.StructureSetROISequence[j].ROIName)
            break
        print('Input ROI is: ')
        ROI_name = raw_input()
        target.append(ROI_name)
    print('------------------------------------')
    return target


def Read_scan(path):  # Read scans under the specified path
    # cts = []
    # fileL = glob.glob(os.path.join(path, '*.dcm'))
    # for file in fileL:
    #     if 'CT' in os.path.basename(file).upper():
    #         cts.append(file)
    # scan = [pydicom.dcmread(s, force=True)
    #         for s in cts]

    scan = [pydicom.dcmread(s, force=True)
            for s in glob.glob(os.path.join(path, 'CT*.dcm'))]
    try:
        # sort slices based on Z coordinate
        scan.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    except:
        print('AttributeError: Cannot read scans')
    return scan


def Read_RTSTRUCT(path):  # Read RTSTRUCT under the specified path
    try:
        rt = [pydicom.dcmread(s, force=True)
              for s in glob.glob(os.path.join(path, 'RS*.dcm'))]
    except:
        print('AttributeError: Cannot read RTSTRUCT')
    return rt

# def readRTDose(path):
#     rt = None
#     for s in glob.glob(os.path.join(path, '*.dcm')):
#         if 'RS' in os.path.basename(s).upper():
#             try:
#                 rt = pydicom.dcmread(s, force=True)
#             except:
#                 print('AttributeError: Cannot read RTSTRUCT')
#     return rt


def poly2mask(vertex_row_coords, vertex_col_coords, shape):  # Mask interpolation
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


# convert to Hounsfield Unit (HU) by multiplying rescale slope and adding intercept
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)  # convert to int16
    # the code below checks if the image has slope and intercept
    # since MRI images often do not provide these
    try:
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
    except AttributeError:
        pass
    else:
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def Img_Bimask(img_path, rtstruct_path, ROI_name):  # generating image array and binary mask
    print('Generating binary mask based on ROI: %s ......' % ROI_name)
    img_vol = Read_scan(img_path)
    mask_vol = Read_RTSTRUCT(rtstruct_path)
    IM = img_vol[0]  # Slices usually have the same basic information including slice size, patient position, etc.
    IM_P = get_pixels_hu(img_vol)
    M = mask_vol[0]
    num_slice = len(img_vol)
    mask = np.zeros([num_slice, IM.Rows, IM.Columns], dtype=np.uint8)
    xres = np.array(IM.PixelSpacing[0])
    yres = np.array(IM.PixelSpacing[1])
    slice_thickness = np.abs(img_vol[1].ImagePositionPatient[2] - img_vol[0].ImagePositionPatient[2])

    ROI_id = match_ROIid(rtstruct_path, ROI_name)
    # Check DICOM file Modality
    if IM.Modality == 'CT' or 'PT':
        for k in range(len(M.ROIContourSequence[ROI_id].ContourSequence)):
            Cpostion_rt = M.ROIContourSequence[ROI_id].ContourSequence[k].ContourData[2]
            for i in range(num_slice):
                if np.int64(Cpostion_rt) == np.int64(
                        img_vol[i].ImagePositionPatient[2]):  # match the binary mask and the corresponding slice
                    sliceOK = i
                    break
            x = []
            y = []
            z = []
            m = M.ROIContourSequence[ROI_id].ContourSequence[k].ContourData
            for i in range(0, len(m), 3):
                x.append(m[i + 1])
                y.append(m[i + 0])
                z.append(m[i + 2])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            x -= IM.ImagePositionPatient[1]
            y -= IM.ImagePositionPatient[0]
            z -= IM.ImagePositionPatient[2]
            pts = np.zeros([len(x), 3])
            pts[:, 0] = x
            pts[:, 1] = y
            pts[:, 2] = z
            a = 0
            b = 1
            p1 = xres
            p2 = yres
            m = np.zeros([2, 2])
            m[0, 0] = img_vol[sliceOK].ImageOrientationPatient[a] * p1
            m[0, 1] = img_vol[sliceOK].ImageOrientationPatient[a + 3] * p2
            m[1, 0] = img_vol[sliceOK].ImageOrientationPatient[b] * p1
            m[1, 1] = img_vol[sliceOK].ImageOrientationPatient[b + 3] * p2
            # Transform points from reference frame to image coordinates
            m_inv = np.linalg.inv(m)
            pts = (np.matmul((m_inv), (pts[:, [a, b]]).T)).T
            mask[sliceOK, :, :] = np.logical_or(mask[sliceOK, :, :],
                                                poly2mask(pts[:, 0], pts[:, 1], [IM_P.shape[1], IM_P.shape[2]]))
    elif IM.Modality == 'MR':
        slice_0 = img_vol[0]
        slice_n = img_vol[-1]

        # the screen coordinates, including the slice number can then be computed
        # using the inverse of this matrix
        transform_matrix = np.r_[slice_0.ImageOrientationPatient[3:], 0, slice_0.ImageOrientationPatient[
                                                                         :3], 0, 0, 0, 0, 0, 1, 1, 1, 1].reshape(4,
                                                                                                                 4).T  # yeah that's ugly but I didn't have enough time to make anything nicer
        T_0 = np.array(slice_0.ImagePositionPatient)
        T_n = np.array(slice_n.ImagePositionPatient)
        col_2 = (T_0 - T_n) / (1 - len(img_vol))
        pix_s = slice_0.PixelSpacing
        transform_matrix[:, -1] = np.r_[T_0, 1]
        transform_matrix[:, 2] = np.r_[col_2, 0]
        transform_matrix[:, 0] *= pix_s[1]
        transform_matrix[:, 1] *= pix_s[0]

        transform_matrix = np.linalg.inv(transform_matrix)
        for s in M.ROIContourSequence[ROI_id].ContourSequence:
            Cpostion_rt = np.r_[s.ContourData[:3], 1]
            roi_slice_nb = int(transform_matrix.dot(Cpostion_rt)[2])
            for i in range(num_slice):
                print(roi_slice_nb, i)
                if roi_slice_nb == i:
                    sliceOK = i
                    break
            x = []
            y = []
            z = []
            m = s.ContourData
            for i in range(0, len(m), 3):
                x.append(m[i + 1])
                y.append(m[i + 0])
                z.append(m[i + 2])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            x -= IM.ImagePositionPatient[1]
            y -= IM.ImagePositionPatient[0]
            z -= IM.ImagePositionPatient[2]
            pts = np.zeros([len(x), 3])
            pts[:, 0] = x
            pts[:, 1] = y
            pts[:, 2] = z
            a = 0
            b = 1
            p1 = xres
            p2 = yres
            m = np.zeros([2, 2])
            m[0, 0] = img_vol[sliceOK].ImageOrientationPatient[a] * p1
            m[0, 1] = img_vol[sliceOK].ImageOrientationPatient[a + 3] * p2
            m[1, 0] = img_vol[sliceOK].ImageOrientationPatient[b] * p1
            m[1, 1] = img_vol[sliceOK].ImageOrientationPatient[b + 3] * p2
            # Transform points from reference frame to image coordinates
            m_inv = np.linalg.inv(m)
            pts = (np.matmul((m_inv), (pts[:, [a, b]]).T)).T
            mask[sliceOK, :, :] = np.logical_or(mask[sliceOK, :, :],
                                                poly2mask(pts[:, 0], pts[:, 1], [IM_P.shape[1], IM_P.shape[2]]))

    # The pixel intensity values are normalized to range [0 255] using linear translation
    IM_P = IM_P.astype(np.float32)
    # IM_P = (IM_P-np.min(IM_P))*255/(np.max(IM_P)-np.min(IM_P))

    Img = sitk.GetImageFromArray(IM_P)  # convert image_array to image
    Mask = sitk.GetImageFromArray(mask)
    # try:
    #     origin = IM.GetOrigin()
    # except:
    #     origin = (0.0, 0.0, 0.0)

    # Set voxel spacing [[pixel spacing_x, pixel spacing_y, slice thickness]
    # slice_thickness = IM.SliceThickness
    Img.SetSpacing([np.float64(xres), np.float64(yres), np.float64(slice_thickness)])
    Mask.SetSpacing([np.float64(xres), np.float64(yres), np.float64(slice_thickness)])

    return Img, Mask

def doseBimask(rtdose, structure, roi):  # generating image array and binary mask
    print('Generating Dose binary mask based on ROI: %d ......' % roi)

    logging.info('Parse RT structure')
    try:
        rtss = [dicomparser.DicomParser(s)
              for s in glob.glob(os.path.join(structure, 'RS*.dcm'))]
    except:
        print('AttributeError: Cannot read RTSTRUCT')
    # rtss = dicomparser.DicomParser(structure)
    structures = rtss[0].GetStructures()
    s = structures[roi]
    s['planes'] = rtss[0].GetStructureCoordinates(roi)
    s['thickness'] = rtss[0].CalculatePlaneThickness(s['planes'])
    planes = s['planes']

    logging.info('Parse RT dose')
    try:
        doseObj = [dicomparser.DicomParser(s)
                    for s in glob.glob(os.path.join(structure, 'RD*.dcm'))]
    except:
        print('AttributeError: Cannot read RTDose')

    # doseObj = dicomparser.DicomParser(rtdose)
    rtDoseM = dose.DoseGrid(doseObj[0].ds)
        # Dose = sitk.GetImageFromArray(rtDoseM.dose_grid)
        # Dose.SetSpacing(rtDoseM.scale)

    if ((len(planes)) and ("PixelData" in doseObj[0].ds)):

        # Get the dose and image data information
        dd = doseObj[0].GetDoseData()
        # id = doseObj.GetImageData()
        
        x,y = np.meshgrid(np.array(dd['lut'][0]), np.array(dd['lut'][1]))
        x,y = x.flatten(),y.flatten()
        dosegridpoints = np.vstack((x, y)).T

    mask = np.zeros(rtDoseM.shape, dtype=np.uint8)
    for z,plane in six.iteritems(planes):
        frame = get_plan_seq(doseObj[0].ds, z, 0.5)
        doseplane = doseObj[0].GetDoseGrid(z)
        contours = [[x[0:2] for x in c['data']] for c in plane]
        if not len(doseplane):
            break
        grid = np.zeros((dd['rows'],dd['columns']),dtype=np.uint8)
        for i, contour in enumerate(contours):
            doselut = dd['lut']
            c = matplotlib.path.Path(list(contour))
            m = c.contains_points(dosegridpoints)
            m = m.reshape((len(doselut[1]),len(doselut[0])))
            grid = np.logical_xor(m.astype(np.uint8),grid)
        mask[:,:,frame] = grid.T
 
        # Slices usually have the same basic information including slice size, patient position, etc.

    # The pixel intensity values are normalized to range [0 255] using linear translation
    rtDoseImg = rtDoseM.dose_grid
    rtDoseImg = rtDoseImg.astype(np.float32)

    #IM_P = (IM_P-np.min(IM_P))*255/(np.max(IM_P)-np.min(IM_P))

    sitkDose = sitk.GetImageFromArray(rtDoseImg)  # convert image_array to image
    sitkMask = sitk.GetImageFromArray(mask)
    # try:
    #     origin = IM.GetOrigin()
    # except:
    #     origin = (0.0, 0.0, 0.0)

    # Set voxel spacing [[pixel spacing_x, pixel spacing_y, slice thickness]
    #slice_thickness = IM.SliceThickness
    sitkDose.SetSpacing(rtDoseM.scale)
    sitkMask.SetSpacing(rtDoseM.scale)

    return sitkDose, sitkMask


def get_plan_seq(ds, z=0, threshold=0.5):
    if 'GridFrameOffsetVector' in ds:
            # pixel_array = self.GetPixelArray()
            z = float(z)
            # Get the initial dose grid position (z) in patient coordinates
            ipp = ds.ImagePositionPatient
            iop = ds.ImageOrientationPatient
            gfov = ds.GridFrameOffsetVector
            # Add the position to the offset vector to determine the
            # z coordinate of each dose plane
            planes = (iop[0] * iop[4] * np.array(gfov)) + ipp[2]
            frame = -1
            # Check to see if the requested plane exists in the array
            if (np.amin(np.fabs(planes - z)) < threshold):
                frame = np.argmin(np.fabs(planes - z))
            # Return the requested dose plane, since it was found
            if not (frame == -1):
                # return pixel_array[frame]
                return frame

            # Check if the requested plane is within the dose grid boundaries
            elif ((z < np.amin(planes)) or (z > np.amax(planes))):
                return np.array([])
            # The requested plane was not found, so interpolate between planes
            else:
                # Determine the upper and lower bounds
                umin = np.fabs(planes - z)
                ub = np.argmin(umin)
                # lmin = umin.copy()
                # Change the min value to the max so we can find the 2nd min
                # lmin[ub] = np.amax(umin)
                # lb = np.argmin(lmin)
                # # Fractional distance of dose plane between upper & lower bound
                # fz = (z - planes[lb]) / (planes[ub] - planes[lb])
                # plane = self.InterpolateDosePlanes(
                #     pixel_array[ub], pixel_array[lb], fz)
                return ub

def main():
    path = '/Users/yang/Downloads/ESO/120115'

    ROI = 'GTV'
    Img, Mask = Img_Bimask(path, path, ROI)

    #pyradiomcis Setting
    params = os.path.join(os.getcwd(), '.', 'examples', 'exampleSettings', 'Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    #extract CT images Radiomics features
    result_1 = extractor.execute(Img, Mask)
    print(result_1)
    # Make an array of the values
    feature_1 = np.array([])

    for key, value in six.iteritems(result_1):
        if key.startswith("original_"):
            feature_1 = np.append(feature_1, result_1[key])

    print(feature_1)

    ROI = 3
    Dose, Mask2 = doseBimask(path,path, ROI)
    result2 = extractor.execute(Dose,Mask2)
    print(result2)
    feature2 = np.array([])

    for key, value in six.iteritems(result2):
        if key.startswith("original_"):
            feature2 = np.append(feature2, result2[key])

    print(feature2)


if __name__ == "__main__":
    main()
