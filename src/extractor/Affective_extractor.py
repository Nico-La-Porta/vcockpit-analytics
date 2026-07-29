import numpy as np
import statsmodels.api as sm

class affective_extractor:
    """
    A class to analyze affective data
    """
    def __init__(self, ONSET_THRESHOLD=0.01, OFFSET_THRESHOLD=0, **kwargs):
        # Instance variables
        self.ONSET_THRESHOLD = ONSET_THRESHOLD  # the threshold in microsiemens required to classify an onset of a peak
        self.OFFSET_THRESHOLD = OFFSET_THRESHOLD  # the threshold in microsiemens required to classify an offset of a peak

        self.variables = kwargs
        self.fs = self.variables['fs']
        self.device = self.variables['device']
        self.RR = self.variables['RR']  # list of RR values
        self.hr = self.variables['hr']  # list of heart rate values
        self.temps_list = self.variables['temps_list']

        '''self.compute_entertainment()
        self.compute_emotional_regulation()
        self.compute_percentage_of_ibi_that_differ()
        self.compute_rmssd()
        self.compute_normal_RR()
        self.compute_emotional_regulation()'''
        
        if self.device != "Bitalino":
            self.eda_fs = self.variables['eda_fs']  # (Hz) Frequency of GSR and Temperaturedata points
            self.MEAN_KERNEL_WIDTH = 5 * self.eda_fs  # The width (data points) of the mean kernel
            self.temps_list = self.variables['temps_list'] # Temperature signal 
            self.eda = self.variables['eda'] # Electro-Dermal Activity (EDA) signal
            '''self.compute_arousal()
            self.compute_stress()
            self.compute_engagement()'''


    ### NTNU Implementations

    def compute_arousal(self):  
        
        """
        calculating arousal based on EDA positive change [1]

        Goes through every data point in a window, sum up all positive changes from subsequent data points

        [1] Leiner, D. J., Fahr, A., & Früh, H. (2012). EDA Positive Change: A Simple Algorithm for
        Electrodermal Activity to Measure General Audience Arousal During Media Exposure.
        Communication Methods and Measures, 6 (4), 237–250.

        :param eda: list of eda data points
        :type eda: list of float
        :return: positive_change - a measure for arousal
        :rtype: list of float
        """
        positive_change = 0
        for i in range(len(self.eda) - 1):
            if self.eda[i + 1] > self.eda[i]:
                positive_change += self.eda[i + 1] - self.eda[i]

        self.positive_change = float(positive_change)
    
    def compute_stress(self):
        """
        Predicts acute stress based on GSR temperature
        :param temps_list: list of temperatures
        :return: returns the overall change in temperature in the list
        """
        slope = np.polyfit([0.25 * i for i in range(len(self.temps_list))], self.temps_list, 1)[0]
        self.negative_slope = float(np.negative(slope))

    def compute_entertainment(self):
        """
        Calculates these features which are correlated with entertainment:
        - The average HRE --> Forse intendevano HR
        - The variance of the HR signal σ2
        - The maximum HR max
        - The minimum HR min
        - The difference D between the maximum and the minimum HR
        - The correlation coefficient R between HR recordings and the time t in which data were recorded
            This parameter provides a notion of the linearity of the signal (HR data) over time
        - The autocorrelation ρ1 (lag equals 1) of the signal, which is used to detect the
            level of non-randomness in the HR data
        - The approximate entropy (ApEnm,r)(Pincus 1991) of the signal which quantifies
            the unpredictability of fluctuations in the HR time series.

        :param hr: list of heart rate values
        :type hr: list of float
        :return: 9 different features as described above
        :rtype: (float, float, float, float, float, float, float, float, float)
        """
        def ApEn(U, m, r) -> float:
            """
            Approximate_entropy. Source:
            https://en.wikipedia.org/wiki/Approximate_entropy
            """

            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

            def _phi(m):
                x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                C = [
                    len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
                    for x_i in x
                ]
                return (N - m + 1.0) ** (-1) * sum(np.log(C))
            N = len(U)
            return abs(_phi(m + 1) - _phi(m))

        self.hr = np.asarray(self.hr)
        avg_hr = np.average(self.hr)
        var_hr = np.var(self.hr)
        max_hr = np.amax(self.hr)
        min_hr = np.amin(self.hr)
        diff = max_hr - min_hr
        p = np.corrcoef(self.hr, np.arange(len(self.hr)))
        p1 = sm.tsa.acf(self.hr, nlags=1, fft=False)
        approximate_entropy = ApEn(self.hr, 2, 3)
        return avg_hr, var_hr, max_hr, min_hr, diff, p1[0], p1[1], approximate_entropy, p[0][1]   

    def compute_percentage_of_ibi_that_differ(self):
        """
        Helper for emotional regulation
        :return: percentage of RR successive RR values that differs by more than 50ms
        :rtype: float
        """
        assert len(self.RR) > 2
        differs_more = 0
        differs_less = 0
        for i in range(1, len(self.RR)):
            if abs(self.RR[i] - self.RR[i - 1]) > 0.05:
                differs_more += 1
            else:
                differs_less += 1
        return max(differs_more / (differs_more + differs_less), 0.01)

    def compute_rmssd(self):
        """
        Helper for emotional regulation
        One way to measure heart rate variability.
        :return: root mean square of successive differences
        :rtype: float
        """
        assert len(self.RR) > 2
        total = 0
        for i in range(1, len(self.RR)):
            total += (self.RR[i] - self.RR[i - 1]) ** 2
        return (total / (len(self.RR) - 1)) ** 0.5

    def compute_normal_RR(self):
        """
        Helper for emotional regulation
        Removes RR values that are below the 10th percentile and above the 90th percentile.
        :return: a list of RR values where the 10th and 90th percentile are removed
        :rtype: list of float
        """

        min_RR = np.percentile(np.asarray(self.RR), 10)
        max_RR = np.percentile(np.asarray(self.RR), 90)
        return [value for value in self.RR if min_RR < value < max_RR]

    def compute_emotional_regulation(self):
        """
        Computes emotional regulation based on a list of RR values
        :return: a measure of emotional regulation
        :rtype: float, float, float
        """
        rmssd = self.compute_rmssd()
        percentage_that_differ = self.compute_percentage_of_ibi_that_differ()
        normal = self.compute_normal_RR()
        return rmssd, percentage_that_differ, np.average(normal)

    def compute_engagement(self):
        """
        Compute three different features correlated with engagement

        :param eda: list of eda data points
        :type eda: list of float
        :return: amplitude, number of peaks of phasic signal, and area under the curve of tonic signal
        :rtype: (float, float, float)
        """

        # find tonic and phasic components
        self.mean_arr = self._mean_filter()
        relevant_eda = self.eda[self.MEAN_KERNEL_WIDTH: - self.MEAN_KERNEL_WIDTH]
        self.tonic = self.mean_arr - abs(min(relevant_eda - self.mean_arr))
        self.phasic = relevant_eda - self.tonic

        # features
        peak_start, peak_end = self._find_peaks(relevant_eda - self.mean_arr)
        self.amplitude = self._find_amplitude(peak_start, peak_end, self.phasic)
        self.nr_peaks = sum(peak_start)
        self.auc = self._area_under_curve(self.tonic)

    def _mean_filter(self):
        """
        Compute the mean eda signal, using a mean kernel of 10 seconds width

        :param eda: list of eda data points
        :type eda: list of float
        :return: mean filter of the signal
        :rtype: np.array
        """
        self.mean_arr = np.array([])
        for i in range(self.MEAN_KERNEL_WIDTH, len(self.eda) - self.MEAN_KERNEL_WIDTH):
            mean = np.mean(self.eda[i - self.MEAN_KERNEL_WIDTH: i + self.MEAN_KERNEL_WIDTH + 1])
            self.mean_arr = np.append(self.mean_arr, mean)

    def _find_peaks(self):
        """
        Find the position of peak start and peak ends on the phasic signal

        :param modified_phasic: list of the phasic signal data points
        :type modified_phasic: list of float
        :return: two lists with 1 where peaks starts and ends respectively
        :rtype: (list of int, list of int)
        """
        self.peak_start = np.zeros(len(self.modified_phasic))
        self.peak_end = np.zeros(len(self.modified_phasic))
        rising = False

        # identify if start is peak
        if self.ONSET_THRESHOLD < self.modified_phasic[0] < self.modified_phasic[1]:
            self.peak_start[0] = 1
            rising = True

        for i in range(len(self.modified_phasic) - 1):
            # peak starts from the point it gets above the onset threshold
            if self.modified_phasic[i] < self.ONSET_THRESHOLD < self.modified_phasic[i + 1] and not rising:
                self.peak_start[i + 1] = 1
                rising = True
            # peak ends from the point it dips below the offset threshold
            elif self.modified_phasic[i] > self.OFFSET_THRESHOLD > self.modified_phasic[i + 1] and rising:
                self.peak_end[i + 1] = 1
                rising = False

    def _find_amplitude(self):
        """
        Find the total amplitude of the highest points of each peak

        :param peak_start: where the peaks in the phasic signal starts
        :type peak_start: list of int
        :param peak_end: where the peaks in the phasic signal ends
        :type peak_end: list of int
        :param phasic: list of the phasic signal data points
        :type phasic: list of float
        :return: the amplitude of the phasic signal
        :rtype: float
        """
        self.amplitude = 0
        for i in range(len(self.phasic)):
            # if we found a peak start
            if self.peak_start[i] == 1:
                j = i
                # find the peak end (or end of data points)
                while j < len(self.phasic) and self.peak_end[j] != 1:
                    j += 1
                # find the highest point of phasic in the range from the start of peak to end of peak
                self.amplitude += max(self.phasic[i:j + 1])

    def _area_under_curve(self):
        """
        Computes the area under the curve of the tonic signal, using trigonometry

        :param tonic: list of the tonic signal
        :type tonic: list of float
        :return: area under the curve of the tonic signal
        :rtype: float
        """
        auc = 0
        for i in range(len(self.tonic) - 1):
            # first data point
            y1 = self.tonic[i]
            # second datapoint
            y2 = self.tonic[i + 1]
            # change in y
            dy = y2 - y1
            # change in x
            dx = 1 / self.fs

            area_square = y1 * dx
            area_triangle = dy * dx / 2
            auc += area_square + area_triangle

        return auc
