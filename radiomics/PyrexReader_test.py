#!/usr/bin/env python
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, getTestCase
import pydicom
import radiomics
import PyrexReader
import os
import sys
import six
import SimpleITK as sitk
import logging
import pandas as pd
pd.set_option('display.width', 120)


def PyRF_Batch(dcm_dir, settings_dir, output_dir):
    kvs = ['90Kv', '120Kv', '140Kv']
    thickness = ['1mm', '2mm', '3mm', '5mm']
    recon_algo = ['sharp', 'smooth', 'stand']
    dirLists = os.listdir(dcm_dir)
    for kv in kvs:
        for thick in thickness:
            for algo in recon_algo:
                dcmDirName = 'CIRS-2_CT_' + kv + '_' + thick + '_' + algo + '_DCM'
                rsDirName = 'CIRS-2_CT_' + kv + '_' + thick + '_' + algo + '_RS'
                if dcmDirName in dirLists and rsDirName in dirLists:
                    img_path = os.path.join(dcm_dir, dcmDirName)
                    rs_path = os.path.join(dcm_dir, rsDirName)
                    rs_dcm = None
                    try:
                        for file in os.listdir(rs_path):
                            if 'dcm' in file:
                                print(file)
                                rs_dcm = pydicom.dcmread(
                                    os.path.join(rs_path, file), force=True)
                        #rs_dcm = [pydicom.dcmread(s, force=True) for s in glob.glob(os.path.join(rs_path, '*.dcm'))]
                    except:
                        print('AttributeError: Cannot read RTSTRUCT')
                    if rs_dcm:
                        for roi in range(len(rs_dcm.StructureSetROISequence)):
                            print(rs_dcm.StructureSetROISequence[roi].ROIName)

                            extractPyradiomicsIntoOneFile(img_path,
                                                           rs_path,
                                                           rs_dcm.StructureSetROISequence[roi].ROIName,
                                                           rs_dcm.StructureSetROISequence[roi].ROINumber,
                                                           settings_dir,
                                                           output_dir)


def extractPyradiomics_test():
    # PyRadiomics_settings
    img_path = './DCM_data/140Kv_200ma_1mm_200slices'
    rs_path = './DCM_data/140Kv_200ma_1mm_200slices_RS'
    ROI_Name = 'GTV-100HU_sharp_1mm'
    Img, Mask = PyrexReader.Img_Bimask(img_path, rs_path, ROI_Name)

    setting_parameters = './PyRF_settings/3D.yaml'

    settingsFile = os.path.join(os.getcwd(), setting_parameters)
    extractor = radiomics.featureextractor.RadiomicsFeaturesExtractor(
        settingsFile)

    featureVector = extractor.execute(Img, Mask, label=1)
    print(len(featureVector))
    for (key, val) in six.iteritems(featureVector):
        print("\t%s,%s" % (key, val))
    print(len(featureVector))

    # print('\nfirstOrder:')
    # firstOrderFeatures = firstorder.RadiomicsFirstOrder(Img, Mask)
    # firstOrderFeatures.enableAllFeatures()
    # firstOrderFeatures.calculateFeatures()
    # for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
    #     print("\t%s,%s" % (key, val))
    #
    # print('\nShape:')
    # shapeFeatures = shape.RadiomicsShape(Img, Mask)
    # shapeFeatures.enableAllFeatures()
    # shapeFeatures.calculateFeatures()
    # for (key, val) in six.iteritems(shapeFeatures.featureValues):
    #     print("\t%s,%s" % (key, val))


def extractPyradiomics(img_path, rs_path, ROI_Name, ROI_Number, settingsDir, outputBaseDir):
    # using SimpleITK parse dcm_image and dcm_Rstruct
    Img, Mask = PyrexReader.Img_Bimask(img_path, rs_path, ROI_Name)

    # different settings(2D,3D)
    SETTINGS_BASE_FILE_NAMES = ['2D', '3D']
    for settingsBaseFileName in SETTINGS_BASE_FILE_NAMES:
        settingsFile = os.path.join(
            settingsDir, settingsBaseFileName + ".yaml")
        extractor = radiomics.featureextractor.RadiomicsFeaturesExtractor(
            settingsFile)

        # for different binwidth
        for binWidth in [5, 10, 15, 20, 40]:
            extractor.settings["binWidth"] = binWidth
            features = []
            print("Extracting features for (image/mask): ")

            featureVector = extractor.execute(Img, Mask)
            # featureVector["study"] = row.study
            # featureVector["series"] = row.series
            # featureVector["canonicalType"] = row.canonicalType
            # featureVector["segmentedStructure"] = row.segmentedStructure
            features.append(featureVector)
            print("------------ DONE ------------")

            # postprocess and check features
            dfFeatures = pd.DataFrame(features)
            print(dfFeatures.shape)

            dfNanSel = dfFeatures[(dfFeatures.isnull().any(axis=1))]
            print(dfNanSel.shape)
            print(dfNanSel.loc[:, dfNanSel.isnull().any()])

            # save to disk
            # outputBaseDir = os.path.join(os.getcwd(), "EvalData")
            if not os.path.exists(outputBaseDir):
                os.mkdir(outputBaseDir)
            pyRF_prefix = "pyRF_" + ROI_Name + '_model_' + \
                settingsBaseFileName + "_bin_" + str(binWidth) + ".csv"
            outFileName = os.path.join(outputBaseDir, pyRF_prefix)

            dfFeatures.to_csv(outFileName, sep=";", index=False)
            print("Features saved to", outFileName)

def extractPyradiomicsIntoOneFile(img_path, rs_path, ROI_Name, ROI_Number, settingsDir, outputBaseDir):
    # using SimpleITK parse dcm_image and dcm_Rstruct
    Img, Mask = PyrexReader.Img_Bimask(img_path, rs_path, ROI_Name)

    imageName = os.path.basename(img_path)

    features = []
    # different settings(2D,3D)
    SETTINGS_BASE_FILE_NAMES = ['2D', '3D']
    for settingsBaseFileName in SETTINGS_BASE_FILE_NAMES:
        settingsFile = os.path.join(
            settingsDir, settingsBaseFileName + ".yaml")
        extractor = radiomics.featureextractor.RadiomicsFeaturesExtractor(
            settingsFile)

        # for different binwidth
        for binWidth in [5, 10, 15, 20, 40]:
            extractor.settings["binWidth"] = binWidth
            # features = []
            print("Extracting features for (image/mask):mode:%s bins:%s " %(settingsBaseFileName,binWidth))

            featureVector = extractor.execute(Img, Mask)
            # featureVector["study"] = row.study
            # featureVector["series"] = row.series
            # featureVector["canonicalType"] = row.canonicalType
            # featureVector["segmentedStructure"] = row.segmentedStructure
            features.append(featureVector)
            print("------------ DONE ------------")

    # postprocess and check features

    dfFeatures = pd.DataFrame(features)
    print(dfFeatures.shape)

    dfNanSel = dfFeatures[(dfFeatures.isnull().any(axis=1))]
    print(dfNanSel.shape)
    print(dfNanSel.loc[:, dfNanSel.isnull().any()])

    # save to disk
    # outputBaseDir = os.path.join(os.getcwd(), "EvalData")
    if not os.path.exists(outputBaseDir):
        os.mkdir(outputBaseDir)
    pyRF_prefix = "pyRF_" + imageName +'_' + ROI_Name + ".csv"
    outFileName = os.path.join(outputBaseDir, pyRF_prefix)

    dfFeatures.to_csv(outFileName, sep=";", index=False)
    print("Features saved to", outFileName)


if __name__ == "__main__":
    workHomeDir = os.getcwd()
    settingDir = os.path.join(workHomeDir, 'PyRF_settings')
    # dcmDataDir = os.path.join(workHomeDir, 'CIRS_2_Kv_thickness')
    dcmDataDir = os.path.join(workHomeDir, 'CIRS_2_Algorith')
    pyRFDataDir = os.path.join(workHomeDir, 'PyRF_data')

    # format stringfor log messages
    # Control the amount of logging stored by setting the level of the logger. N.B. if ˓→the level is higher than the
    # Verbositiy level, the logger level will also determine the amount of information ˓→printed to the output
    log_file = os.path.join(workHomeDir, 'PyRF.log')
    handler = logging.FileHandler(filename=log_file, mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    radiomics.logger.setLevel(logging.DEBUG)

    PyRF_Batch(dcmDataDir, settingDir, pyRFDataDir)

    #
    # extractPyradiomics_test()
    # extractPyradiomics(Img, Mask, ROI)
