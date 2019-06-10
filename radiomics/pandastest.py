import pandas as pd
import os,sys

if __name__ == "__main__":
    basepath = '/Users/yang/mybin/005Radiomics/Pyrex/PyRF_data/'
    csvfilelist = os.listdir(basepath)
    print(csvfilelist[0])
    data = pd.read_csv(os.path.join(basepath, csvfilelist[0]), header=0)
    print('here')