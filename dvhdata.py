import os
import sys
from dicompylercore.dvh import DVH


class dvhdata(DVH):
    """class that modify classs DVH from dicompylercore"""

    def __init__(self, dvh):
        self.counts = dvh.counts
        self.bins = dvh.bins
        self.dvh_type = dvh.dvh_type
        self.dose_units = dvh.dose_units
        self.volume_units = dvh.volume_units
        self.rx_dose = dvh.rx_dose
        self.name = dvh.name
        self.color = dvh.color
        self.notes = dvh.notes
        # DVH.__init__(self,counts,bins,dvh_type,dose_units,volume_units,rx_dose,name,color,notes)

    def getDifferences(self, dvh, result):
        """compare dvh with another dvh, compute the difffereces"""
        fileObj = open(result, 'a')

        if not (self.dose_units == dvh.dose_units) or \
                not (self.volume_units == dvh.volume_units):
            raise AttributeError("DVH units are not equivalent")

        def fmtcmp(attr, units, ref=self, comp=dvh):
            """Generate arguments for string formatting.

            Parameters
            ----------
            attr : string
                Attribute used for comparison
            units : string
                Units used for the value

            Returns
            -------
            tuple
                tuple used in a string formatter
                use: cGy for difference
            """
            if attr in ['volume', 'max', 'min', 'mean']:
                val = ref.__getattribute__(attr) * 100
                cmpval = comp.__getattribute__(attr) * 100
            else:
                val = ref.statistic(attr).value * 100
                cmpval = comp.statistic(attr).value * 100
            return attr.capitalize() + ":", val, units, cmpval, units, \
                0 if not val else ((cmpval - val) / val) * 100, cmpval - val

        def savefmtcmp(attr, units, ref=self, comp=dvh):
            """Generate arguments for string formatting.
            """
            if attr in ['volume', 'max', 'min', 'mean']:
                val = ref.__getattribute__(attr)
                cmpval = comp.__getattribute__(attr)
            else:
                val = ref.statistic(attr).value
                cmpval = comp.statistic(attr).value
            # strValue = str(comp.__getattribute__('notes') + ',' + comp.__getattribute__(
            #     'name') + ',' + attr.capitalize() + "," + str(val) + "," + units +
            #                "," + str(cmpval) + "," + units + "," + str(
            #     0 if not val else ((cmpval - val) / val) * 100) + "," + str(cmpval - val) + "," + units + '\n')
            strValue = str(comp.__getattribute__(
                'name') + ',' + attr.capitalize() + "," + str(val) +
                "," + str(cmpval) + "," + str(
                0 if not val else ((cmpval - val) / val) * 100) + "," + str(cmpval - val) + '\n')

            return strValue

        print("{:11} {:>14} {:>17} {:>17} {:>14}".format(
            'Structure:', self.name, dvh.name, 'Rel Diff', 'Abs diff'))
        print("-----")
        dose = "rel dose" if self.dose_units == '%' else \
            "abs dose: {}".format(self.dose_units)
        vol = "rel volume" if self.volume_units == '%' else \
            "abs volume: {}".format(self.volume_units)
        print("DVH Type:  {}, {}, {}".format(self.dvh_type, dose, vol))
        fmtstr = "{:11} {:12.2f} {:3}{:14.2f} {:3}{:+14.2f} % {:+14.2f}"
        print(fmtstr.format(*fmtcmp('volume', self.volume_units)))
        print(fmtstr.format(*fmtcmp('max', self.dose_units)))
        print(fmtstr.format(*fmtcmp('min', self.dose_units)))
        print(fmtstr.format(*fmtcmp('mean', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D100', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D98', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D95', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D90', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D50', self.dose_units)))
        # Only show volume statistics if a Rx Dose has been defined
        # i.e. dose is in relative units
        if self.dose_units == '%':
            print(fmtstr.format(
                *fmtcmp('V100', self.dose_units,
                        self.relative_dose(), dvh.relative_dose())))

            print(fmtstr.format(
                *fmtcmp('V95', self.dose_units,
                        self.relative_dose(), dvh.relative_dose())))

            print(fmtstr.format(
                *fmtcmp('V5', self.dose_units,
                        self.relative_dose(), dvh.relative_dose())))

            fileObj.write((savefmtcmp('V100', self.dose_units,
                                      self.relative_dose(), dvh.relative_dose())))
            fileObj.write((savefmtcmp('V95', self.dose_units,
                                      self.relative_dose(), dvh.relative_dose())))
            fileObj.write(
                (savefmtcmp('V5', self.dose_units, self.relative_dose(), dvh.relative_dose())))
        print(fmtstr.format(*fmtcmp('D2cc', self.dose_units)))
        # print(self.volume_constraint(20, 'Gy'))
        # print(dvh.volume_constraint(20, 'Gy'))

        fileObj.write(savefmtcmp('volume', self.dose_units))
        fileObj.write(savefmtcmp('max', self.dose_units))
        fileObj.write(savefmtcmp('min', self.dose_units))
        fileObj.write(savefmtcmp('mean', self.dose_units))
        # fileObj.write(savefmtcmp('D100', self.dose_units))
        fileObj.write(savefmtcmp('D98', self.dose_units))
        fileObj.write(savefmtcmp('D95', self.dose_units))
        fileObj.write(savefmtcmp('D90', self.dose_units))
        fileObj.write(savefmtcmp('D50', self.dose_units))
        fileObj.write(savefmtcmp('D2cc', self.dose_units))
        fileObj.close()

        # self.plot()
        # dvh.plot()

    def compare(self, dvh):
        """Compare the DVH properties with another DVH.

        Parameters
        ----------
        dvh : DVH
            DVH instance to compare against

        Raises
        ------
        AttributeError
            If DVHs do not have equivalent dose & volume units
        """
        if not (self.dose_units == dvh.dose_units) or \
                not (self.volume_units == dvh.volume_units):
            raise AttributeError("DVH units are not equivalent")

        def fmtcmp(attr, units, ref=self, comp=dvh):
            """Generate arguments for string formatting.

            Parameters
            ----------
            attr : string
                Attribute used for comparison
            units : string
                Units used for the value

            Returns
            -------
            tuple
                tuple used in a string formatter
            """
            if attr in ['volume', 'max', 'min', 'mean']:
                val = ref.__getattribute__(attr)
                cmpval = comp.__getattribute__(attr)
            else:
                val = ref.statistic(attr).value
                cmpval = comp.statistic(attr).value
            return attr.capitalize() + ":", val, units, cmpval, units, \
                0 if not val else ((cmpval - val) / val) * 100, cmpval - val

        print("{:11} {:>14} {:>17} {:>17} {:>14}".format(
            'Structure:', self.name, dvh.name, 'Rel Diff', 'Abs diff'))
        print("-----")
        dose = "rel dose" if self.dose_units == '%' else \
            "abs dose: {}".format(self.dose_units)
        vol = "rel volume" if self.volume_units == '%' else \
            "abs volume: {}".format(self.volume_units)
        print("DVH Type:  {}, {}, {}".format(self.dvh_type, dose, vol))
        fmtstr = "{:11} {:12.2f} {:3}{:14.2f} {:3}{:+14.2f} % {:+14.2f}"
        print(fmtstr.format(*fmtcmp('volume', self.volume_units)))
        print(fmtstr.format(*fmtcmp('max', self.dose_units)))
        print(fmtstr.format(*fmtcmp('min', self.dose_units)))
        print(fmtstr.format(*fmtcmp('mean', self.dose_units)))
        # print(fmtstr.format(*fmtcmp('D100', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D98', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D95', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D90', self.dose_units)))
        print(fmtstr.format(*fmtcmp('D50', self.dose_units)))
        # Only show volume statistics if a Rx Dose has been defined
        # i.e. dose is in relative units
        if self.dose_units == '%':
            print(fmtstr.format(
                *fmtcmp('V100', self.dose_units,
                        self.relative_dose(), dvh.relative_dose())))
            print(fmtstr.format(
                *fmtcmp('V95', self.dose_units,
                        self.relative_dose(), dvh.relative_dose())))
            print(fmtstr.format(
                *fmtcmp('V5', self.dose_units,
                        self.relative_dose(), dvh.relative_dose())))
        print(fmtstr.format(*fmtcmp('D2cc', self.dose_units)))

        self.plot()
        dvh.plot()

    def plot(self):
        """Plot the DVH using Matplotlib if present."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('Matplotlib could not be loaded. Install and try again.')
        else:
            plt.plot(self.bincenters, self.counts, label=self.name,
                     color=None if not isinstance(self.color, np.ndarray) else
                     (self.color / 255))
            plt.axis([0, 70, 0, 80])  # for relative volume
            plt.xlabel('Dose [%s]' % self.dose_units)
            plt.ylabel('Volume [%s]' % self.volume_units)
            if self.name:
                plt.legend(loc='best')
            plt.grid(True)
            plt.show()
        return self

    def cal_nrmsd(self, dvh, result):

        fileObj = open(result, 'a')

        h1 = self.bins
        h2 = dvh.bins

        rms = np.sqrt(reduce(operator.add, map(
            lambda a, b: (a - b) ** 2, h1, h2)) / len(h1))
        value = dvh.notes + ',' + dvh.name + ',' + str(rms) + '\n'
        fileObj.write(value)
        fileObj.close()
        print(rms)
    # def OAR_constans(self,abs_dose):
    #     self.cumulative
    #     self.
