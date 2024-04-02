from __future__ import annotations
import numpy as np
from typing import List, Union, Optional
import spectral_constants


class Spectrum:
    def __init__(
        self,
        wavelengths: Union[np.ndarray, List[float]],
        amplitudes: Union[np.ndarray, List[float]],
    ):
        """
        Create a Spectrum object. Radiometrc and photometric calculations assume amplitudes data in units of [uW/cm^2/nm] (absolute spectral irradiance).
        NOTE: method comments to follow assume standard units of uW/cm^2/nm.

        Parameters:
        wavelengths (list or np.array): abscissa values for spectrum amplitudes
        amplitudes (list or np.array): spectrum data
        """
        if len(wavelengths) != len(amplitudes):
            raise SystemError("Length of wavelength and amplitude vectors must match.")

        if not isinstance(wavelengths, np.ndarray):
            self.wavelengths = np.array(wavelengths, float)
        else:
            self.wavelengths = wavelengths

        positive = np.all(self.wavelengths > 0)
        sorted_ascending = (np.diff(self.wavelengths) >= 0).all()
        if not positive or not sorted_ascending:
            raise SystemError(
                "Spectral wavelengths must be a vector of values in ascending order, where each value is greater than 0."
            )

        if self.wavelengths.size != np.unique(self.wavelengths).size:
            raise SystemError("All wavelength values must be unique.")

        if not isinstance(amplitudes, np.ndarray):
            self.amplitudes = np.array(amplitudes, float)
        else:
            self.amplitudes = amplitudes

        if not np.isrealobj(self.wavelengths) and np.isrealobj(self.amplitudes):
            raise TypeError(
                "Spectral wavelength and amplitude arrays must be numeric only."
            )

    def __len__(self) -> int:
        return self.wavelengths.size

    def __getitem__(self, index) -> tuple:
        return (self.wavelengths[index], self.amplitudes[index])

    def __str__(self) -> str:
        if len(self) == 1:
            return f"Spectrum(wavelength={self.wavelengths[0]}, amplitude={self.amplitudes[0]})"
        return f"Spectrum(wavelengths={self.wavelengths[0]}... {self.wavelengths[-1]}, amplitudes={self.amplitudes[0]}... {self.amplitudes[-1]})"

    def __mul__(self, s: Union[Spectrum, float, int]) -> Spectrum:
        if isinstance(s, Spectrum):
            if not np.array_equal(self.wavelengths, s.wavelengths):
                raise ValueError("Spectra must share an abscissa to be multiplied.")
            return Spectrum(
                self.wavelengths, np.multiply(self.amplitudes, s.amplitudes)
            )
        if isinstance(s, float) or isinstance(s, int):
            return Spectrum(self.wavelengths, self.amplitudes * s)
        raise TypeError(
            "Spectrum multiplication requires Spectrum objects, or a Spectrum and numeric type."
        )

    # TODO add appropriate dunder to slice spectral object, e.g., spectrum[i:k] slices amplitude and wavelength indices appropriately
    # TODO operator overloading for whole vector arithmetic?

    def wavelength_indices(
        self, lower_wavelength, upper_wavelength
    ) -> tuple[float, float]:
        """
        Returns indice(s) associated with wavelength pair.

        Parameters:
            lower_wavelength: the starting point of the wavelength interval
            upper_wavelength: the ending point of the wavelength interval

        Returns:
            tuple: index pair associated with lower_wavelength and upper_wavelength
        """
        if lower_wavelength >= upper_wavelength:
            raise IndexError(
                "Lower wavelength argument must not be equal or larger than upper."
            )
        return (
            np.where(self.wavelengths == lower_wavelength)[0][0],
            np.where(self.wavelengths == upper_wavelength)[0][0],
        )

    def peak_amplitudes(
        self, interval=[0, np.Inf], return_as_list=False, indices=False
    ) -> Union[np.ndarray, List[float]]:
        """
        Find peak amplitudes in spectral data.

        Parameters:
            interval (sequence, optional): A two-element sequence specifying the minimum value and maximum value for returned peak(s).
            Default is [0,np.Inf]

        Returns:
            list or np.array: Peak values found in amplitude data, given the specified interval.
        """
        from scipy.signal import find_peaks

        if not isinstance(interval, (list, tuple, np.ndarray)) or len(interval) != 2:
            raise ValueError("Input data must be a 2-element sequence.")
        if np.isnan(interval).any() or np.isinf(interval).any():
            raise ValueError("Input data contains NaN or Inf values.")
        if interval[1] <= interval[0]:
            raise ValueError("Second element must be greater than the first element.")

        peaks, _ = find_peaks(self.amplitudes, height=interval)
        if return_as_list:
            if indices:
                return peaks.tolist()
            return self.amplitudes[peaks].tolist()
        if indices:
            return peaks
        return self.amplitudes[peaks]

    def peak_indices(
        self, interval=[0, np.Inf], return_as_list=False
    ) -> Union[np.ndarray, List[int]]:
        """
        Find peak indices in spectral data.
        See #peak_amplitudes

        Returns:
            list or np.array: Peak indices corresponding to amplitude data, given the specified interval.
        """
        return self.peak_amplitudes(interval, return_as_list, indices=True)

    def full_width_half_max(self, i: int) -> float:
        """
        Find full width half max (FWHM) in abscissa units for a given amplitude index.

        Parameters:
            i (int): index where peak exists

        Returns:
            float: FWHM for a given amplitude peak index.

        Warnings
        --------
        PeakPropertyWarning
            If `v` is not actually a peak index. To guard against this, pass index values returned from #peak_indices.
        """
        if not 0 <= i <= len(self.amplitudes) - 1:
            raise IndexError(
                f"Index {i} does not exist! Param v must be an index between [0, {len(self.wavelengths)-1}]."
            )
        if not isinstance(i, (int, np.integer)):
            raise TypeError(
                "Param v must be an integer index associated with a peak in amplitudes data."
            )

        from scipy.signal import peak_widths

        p = peak_widths(self.amplitudes, peaks=[i], rel_height=0.5)
        left = round(p[2][0])
        right = round(p[3][0])
        return self.wavelengths[right] - self.wavelengths[left]

    def is_saturated(self, index_tolerance=3) -> bool:
        """
        Determine if a spectrum is saturated, defined as n indices of the same value, where those n indices are necessarily the maximum value.

        Parameters:
        index_tolerance (int, optional): Some number of indices n that, if those indices possess identical values, qualify a spectrum as saturated.
            Default is 3.

        Returns:
            bool: if a spectrum is saturated.
        """
        max_value = max(self.amplitudes)
        count = 0
        for i in range(len(self.amplitudes)):
            if self.amplitudes[i] == max_value:
                count += 1
                if count >= index_tolerance:
                    return True
            else:
                count = 0
        return False

    def compute_microwatts_per_sq_cm(
        self, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
    ) -> float:
        """
        Compute irradiance in [uW/cm^2]. Calculation assumes spectral data in units of [uW/cm^2/nm] (absolute spectral irradiance).

        Parameters:
        lower_bound (float): lower integration bound in abscissa units.
            Default is smallest amplitude value (0th index).
        upper_bound (float): upper integration bound in abscissa units.
            NOTE: lower_bound arguments which do not exactly match an abscissa value are rounded up to the nearest value.
            E.g., only those abscissa values greater than or equal to the specified lower_bound are included in the integration interval.
            The same is true for upper_bound; only those abscissa values less than the upper_bound are included in the interval.
            Default is largest amplitude value (-1st index).

        Returns:
            float: irradiance
        """
        return self._trapezoidal_integration(
            lower_bound=lower_bound, upper_bound=upper_bound
        )

    def _trapezoidal_integration(
        self, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
    ) -> float:
        """
        Numpy trapezoidal integration wrapper for Spectrum objects.
        """
        lower_bound = self.wavelengths[0] if lower_bound is None else lower_bound
        upper_bound = self.wavelengths[-1] if upper_bound is None else upper_bound

        if lower_bound < self.wavelengths[0]:
            raise IndexError(
                "lower_bound is out of bounds. Specified value is less than smallest amplitude value."
            )
        if upper_bound > self.wavelengths[-1]:
            raise IndexError(
                "upper_bound out of bounds. Specified value is greater than the maximum amplitude value."
            )
        if lower_bound >= upper_bound:
            raise IndexError(
                "Lower wavelength argument must not be equal or larger than upper."
            )

        mask = (self.wavelengths >= lower_bound) & (self.wavelengths <= upper_bound)
        x = self.wavelengths[mask]
        y = self.amplitudes[mask]

        return np.trapz(y=y, x=x)

    def compute_microwatts(
        self,
        collection_area: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> float:
        """
        Compute absolute power in uW.

        Parameters:
        collection_area: incident area for source in cm^2.
        lower_bound, upper_bound: see #compute_microwatts_per_cm

        Returns:
            float: absolute power in uW.
        """
        if collection_area < 0:
            raise ValueError("Collection area cannot be less than 0 cm^2.")
        return (
            self.compute_microwatts_per_sq_cm(
                lower_bound=lower_bound, upper_bound=upper_bound
            )
            * collection_area
        )

    def compute_microwatts_per_nm(self, collection_area: Union[float, int]) -> Spectrum:
        """
        Compute spectral power in uW/nm.

        Parameters:
        collection_area: see #compute_microwatts

        Returns:
            float: spectral power in uW/nm
        """
        return self * collection_area

    def compute_joules(
        self,
        collection_area: float,
        integration_time: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> float:
        """
        Compute energy in Joules.

        Parameters:
        integration_time: temporal integration period in seconds.
        collection_area, lower_bound, upper_bound: see #compute_microwatts

        Returns:
            float: total energy in Joules.
        """
        if integration_time < 0:
            raise ValueError("Integration time cannot be less than 0s.")
        return (
            self.compute_microwatts(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                collection_area=collection_area,
            )
            * integration_time
            * 1e-6
        )

    def compute_electron_volts(
        self,
        collection_area: float,
        integration_time: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> float:
        """
        Compute energy in electron volts.

        Parameters:
            See #compute_joules.

        Returns:
            float: total energy in electron volts.
        """
        return self.compute_joules(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            collection_area=collection_area,
            integration_time=integration_time,
        ) * (1 / spectral_constants.ELECTRON_CHARGE)

    def compute_lux(
        self, luminous_efficiency: Spectrum, max_luminous_efficiency_coefficient: float
    ) -> float:
        """
        Compute lux in lumen/m^2 for an arbitrary luminous_efficiency function.

        Parameters:
            luminous_efficiency (Spectrum): arbitrary luminous efficiency spectrum give as Spectrum object.

        Returns:
            float: lux value
        """
        microwatt_to_watt = 1e-6
        per_cm_sq_to_m_sq = 1e-4
        watt_per_m_sq_per_nm = self * microwatt_to_watt * per_cm_sq_to_m_sq
        lumen_per_m_sq_per_nm = watt_per_m_sq_per_nm * luminous_efficiency
        return (
            max_luminous_efficiency_coefficient
            * lumen_per_m_sq_per_nm._trapezoidal_integration()
        )

    def compute_lux_photopic(self) -> float:
        """
        Compute lux in lumen/m^2 for the photopic luminous efficiency spectrum.

        Returns:
            float: lux value
        """
        photopic = Spectrum(
            spectral_constants.EFFICIENCY_FUNCTION_WAVELENGTHS,
            spectral_constants.PEAK_PHOTOPIC_LUMINOSITY,
        )
        return self.compute_lux(photopic, spectral_constants.PEAK_PHOTOPIC_LUMINOSITY)


# /**
#  * Compute photopic lux in [lumen/m^2]. Calculation assumes #getSpectralData in units of [uW/cm^2/nm] (absolute spectral irradiance).
#  * See base method #computeLux.
# */
# double Spectrum::computeLuxPhotopic( ) {
#     Spectrum photopic { photopicEfficiencyData, efficiencyFunctionWavelengths };
#     return computeLux( photopic, peakPhotopicLuminosity );
# }

# /**
#  * Compute scotopic lux in [lumen/m^2]. Calculation assumes #getSpectralData in units of [uW/cm^2/nm] (absolute spectral irradiance).
#  * See base method #computeLux.
# */
# double Spectrum::computeLuxScotopic( ) {
#     Spectrum scotopic { scotopicEfficiencyData, efficiencyFunctionWavelengths };
#     return computeLux( scotopic, peakScotopicLuminosity );
# }

# /**
#  * Compute lumens. Calculation assumes #getSpectralData in units of [uW/cm^2/nm] (absolute spectral irradiance).
#  * collectionArea in [cm^2].
# */
# double Spectrum::computeLumens( Spectrum & luminousEfficiencyFunction,
#                                 double & maxLuminousEfficiencyCoefficient,
#                                 double & collectionArea ) {
#     double convertPerMeterSquaredToPerCentimeterSquared = 10000.0;
#     return collectionArea * convertPerMeterSquaredToPerCentimeterSquared * computeLux( luminousEfficiencyFunction, maxLuminousEfficiencyCoefficient );
# }

# /**
#  * Compute candella in [lumen/steradians]. Calculation assumes #getSpectralData in units of [uW/cm^2/nm] (absolute spectral irradiance).
#  * Calculation replicates Ocean Optics Spectral Processing and Math AdvancedPhotometrics#computeLuminousIntensityCandela.
#  * solidAngle in [steradians]. collectionArea in [cm^2].
# */
# double Spectrum::computeCandella(   Spectrum & luminousEfficiencyFunction,
#                                     double & maxLuminousEfficiencyCoefficient,
#                                     double & collectionArea,
#                                     double & solidAngle ) {
#     return computeLumens( luminousEfficiencyFunction, maxLuminousEfficiencyCoefficient, collectionArea ) / solidAngle;
# }

# /**
#  * Compute luminous flux in [photons/cm^2/s]. Calculation assumes #getSpectralData in units of [uW/cm^2/nm] (absolute spectral irradiance).
#  * Calculation replicates Ocean Optics Spectral Processing and Math AdvancedPhotometrics#computePhotonsPerCmSquaredPerSecond.
# */
# double Spectrum::computePhotonsPerCmSquaredPerSecond( int & lowerBoundWavelength, int & upperBoundWavelength ) {
#     std::vector<double> photonsPerCmSquaredPerSecondPerNanometer = {};
#     double plancksConstant = 6.62607015e-34;
#     double velocityOfLight = 2.998e8;
#     double metersToNanometer = 1e-9;
#     double wattsToMicrowatts = 1e-6;
#     for ( int i = 0; i < getSpectralData().size(); i++ ) {
#         photonsPerCmSquaredPerSecondPerNanometer.push_back( (getSpectralData().at(i) * getSpectralWavelengths().at(i) * metersToNanometer * wattsToMicrowatts) / ( plancksConstant * velocityOfLight ) );
#     }
#     return Spectrum{ photonsPerCmSquaredPerSecondPerNanometer, getSpectralWavelengths() }.computeIntegral( lowerBound, upperBound );
# }

# /**
#  * Compute total photons. Calculation assumes #getSpectralData in units of [uW/cm^2/nm] (absolute spectral irradiance).
#  * Calculation replicates Ocean Optics Spectral Processing and Math AdvancedPhotometrics#computeTotalPhotons.
#  * integrationTime in [seconds]. collectionArea in [cm^2].
# */
# double Spectrum::computeTotalPhotons( int & lowerBoundWavelength, int & upperBoundWavelength, double & collectionArea, double & integrationTime ) {
#     return computePhotonsPerCmSquaredPerSecond( lowerBoundWavelength, upperBoundWavelength ) * collectionArea * integrationTime;
# }


#
# test and temp content
#


def plot(s: Spectrum):
    import matplotlib.pyplot as plt

    plt.plot(s.wavelengths, s.amplitudes)
    plt.show()


# amplitudes = [0.027, 0.082, 0.082, 0.27, 0.826, 2.732, 7.923, 15.709, 25.954, 40.98, 62.139, 94.951, 142.078, 220.609, 303.464, 343.549, 484.93, 588.746, 651.582, 679.551, 683, 679.585, 650.216, 594.21, 517.031, 430.973, 343.549, 260.223, 180.995, 119.525, 73.081, 41.663, 21.856, 11.611, 5.607, 2.802, 1.428, 0.715, 0.355, 0.17, 0.082, 0.041, 0.02]
# wavelengths = [380, 390, 391, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 507, 510, 520, 530, 540, 550, 555, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770]
wavelengths = np.linspace(1 / 1000, np.pi, 1000)
amplitudes = np.sin(20 * wavelengths) * np.sin(wavelengths)
# amplitudes[34] = .8
# amplitudes[35] = .8
# amplitudes[36] = .8

s = Spectrum(wavelengths, amplitudes)
print("String of 0th index: " + str(s[0]))
print("Length: " + str(len(s)))
print("String of object: " + str(s))
p = s.peak_indices([0.5, 1])
print("Peaks: " + str(p))
print(f"0th peak index={p[0]}, value={amplitudes[p[0]]}")
print("FWHM: " + str(s.full_width_half_max(p[0])))
print("saturated? " + str(s.is_saturated()))
print(
    "irradiance: " + str(s.compute_microwatts_per_sq_cm(lower_bound=1.2, upper_bound=3))
)
plot(s * 3.13)
