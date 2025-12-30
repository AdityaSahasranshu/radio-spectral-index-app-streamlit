# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:53:30 2025

@author: adity
"""

import streamlit as st
import numpy as np
import warnings
import os
import tempfile
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import mad_std
from radio_beam import Beam
from reproject import reproject_interp
import matplotlib.pyplot as plt

# Web page setup
st.set_page_config(page_title="Spectral Index Tool", layout="wide")
warnings.filterwarnings('ignore')

class RadioMap:
    def __init__(self, uploaded_file, name="Map"):
        self.name = name
        # --- WEB ADAPTATION: Handle file object instead of path string ---
        self.tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
        self.tfile.write(uploaded_file.getvalue())
        self.tfile.close()
        self.filepath = self.tfile.name
        # ---------------------------------------------------------------
        
        self.data = None
        self.header = None
        self.wcs = None
        self.beam = None
        self.freq = None
        
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.filepath):
            st.error(f"File not found: {self.filepath}") # Changed raise to st.error
            st.stop()
        
        with fits.open(self.filepath) as hdul:
            hdu = hdul[0] if len(hdul) > 0 and hdul[0].data is not None else hdul[1]
            self.header = hdu.header
            self.data = np.squeeze(hdu.data)
            self.wcs = WCS(self.header).celestial
            
        st.write(f"**Loaded {self.name}**") # Changed print to st.write
        self._get_frequency()
        self._get_beam()
        self._check_units()

    def _get_frequency(self):
        keys = ['RESTFRQ', 'FREQ', 'CRVAL3'] 
        found = False
        for k in keys:
            if k in self.header:
                val = self.header[k]
                if val > 1e6:
                    self.freq = val * u.Hz
                    st.success(f"  Freq: {self.freq.to(u.MHz):.2f}") # Changed print to st.success
                    found = True
                    break
        if not found:
            # --- TRANSLATION START: Replaced input() with st.number_input() ---
            st.warning(f"Frequency missing for {self.name}")
            val = st.number_input(f"Enter Frequency for {self.name} (MHz):", value=0.0, step=1.0, key=f"freq_{self.name}")
            
            # This logic mimics input(): it pauses the code until user types a value > 0
            if val == 0.0:
                st.info("Waiting for input...")
                st.stop() 
            
            self.freq = val * u.MHz
            # --- TRANSLATION END ---

    def _get_beam(self):
        try:
            self.beam = Beam.from_fits_header(self.header)
            st.info(f"  Beam: {self.beam}") # Changed print to st.info
        except:
            # --- TRANSLATION START: Replaced input() with st.number_input() ---
            st.warning(f"Beam info missing in {self.name}.")
            
            c1, c2, c3 = st.columns(3)
            bmaj = c1.number_input(f"{self.name} Major Axis (arcsec):", value=0.0, key=f"bmaj_{self.name}")
            bmin = c2.number_input(f"{self.name} Minor Axis (arcsec):", value=0.0, key=f"bmin_{self.name}")
            bpa = c3.number_input(f"{self.name} PA (deg):", value=0.0, key=f"bpa_{self.name}")
            
            # Pause until user fills all 3
            if bmaj == 0.0 or bmin == 0.0: 
                st.info("Waiting for beam inputs...")
                st.stop()

            self.beam = Beam(major=bmaj*u.arcsec, minor=bmin*u.arcsec, pa=bpa*u.deg)
            # --- TRANSLATION END ---

    def _check_units(self):
        """Detects mJy vs Jy and converts to Jy if needed."""
        unit_str = self.header.get('BUNIT', '').lower()
        
        if 'mjy' in unit_str:
            st.write("  [Correction] Converting mJy -> Jy")
            self.data = self.data / 1000.0
            self.header['BUNIT'] = 'Jy/beam'
        elif 'jy' in unit_str:
            pass 
        else:
            # --- TRANSLATION START ---
            st.warning("Unit unknown. Is this map in mJy/beam?")
            is_mjy = st.radio(f"Units for {self.name}", ["Select...", "Yes (mJy)", "No (Jy)"], key=f"unit_{self.name}")
            
            if is_mjy == "Select...":
                st.stop() # Pause
            
            if is_mjy == "Yes (mJy)":
                self.data = self.data / 1000.0
            # --- TRANSLATION END ---

    def cleanup(self):
        os.unlink(self.filepath)

def convolve_to_common(source_map, target_beam):
    # Check if convolution is needed
    if source_map.beam == target_beam:
        st.write(f"  {source_map.name} already matches target beam.")
        return source_map.data

    st.write(f"--- Convolving {source_map.name} ---")
    
    try:
        # 1. Calculate Kernel
        kernel = target_beam.deconvolve(source_map.beam)
        kernel_pix = kernel.as_kernel(source_map.wcs.proj_plane_pixel_area()**0.5)
        
        # 2. Convolve (FFT)
        from astropy.convolution import convolve_fft
        convolved_data = convolve_fft(source_map.data, kernel_pix, allow_huge=True)
        
        # 3. Apply Beam Area Scaling
        source_area = source_map.beam.sr.value
        target_area = target_beam.sr.value
        scale_factor = target_area / source_area
        
        return convolved_data * scale_factor
        
    except ValueError:
        st.error("  [CRITICAL ERROR] Deconvolution failed.")
        return source_map.data

# --- MAIN WORKFLOW (Replaces 'if __name__ == "__main__":') ---
def calculate_spectral_index_workflow():
    st.title("Spectral Index Pipeline")

    # 1. INPUTS (Replaces path = input())
    col1, col2 = st.columns(2)
    f1 = col1.file_uploader("Upload FITS File 1", type=["fits"])
    f2 = col2.file_uploader("Upload FITS File 2", type=["fits"])

    if f1 and f2:
        # The class initiates and runs its own checks (Freq/Beam/Units)
        # If anything is missing, the class methods above will PAUSE the script (st.stop)
        # The script only reaches line 160 if the user has provided all data.
        map1 = RadioMap(f1, "Map 1")
        map2 = RadioMap(f2, "Map 2")

        # 2. DEFINE COMMON BEAM
        max_axis_1 = max(map1.beam.major, map1.beam.minor)
        max_axis_2 = max(map2.beam.major, map2.beam.minor)
        common_size = max(max_axis_1, max_axis_2) * 1.01 
        common_beam = Beam(major=common_size, minor=common_size, pa=0*u.deg)
        st.write(f"**Common Beam:** {common_beam.major.to(u.arcsec):.2f}")

        # 3. CONVOLVE
        data1_conv = convolve_to_common(map1, common_beam)
        data2_conv = convolve_to_common(map2, common_beam)

        # 4. REGRIDDING
        st.write(f"--- Regridding Map 1 to Map 2 Grid ---")
        data1_aligned, footprint = reproject_interp(
            (data1_conv, map1.wcs),
            map2.wcs,
            shape_out=map2.data.shape,
            order=3
        )

        # 5. NOISE & MASKING
        rms_1 = mad_std(data1_aligned, ignore_nan=True)
        rms_2 = mad_std(data2_conv, ignore_nan=True)
        st.write(f"RMS 1: {rms_1:.4e} | RMS 2: {rms_2:.4e}")

        # --- TRANSLATION: Replaces input() for sigma ---
        sigma_thresh = st.number_input("Enter Sigma Threshold:", value=0.0, step=0.5)
        if sigma_thresh == 0.0:
            st.info("Enter a Sigma Threshold to continue...")
            st.stop()
        # -----------------------------------------------

        mask = (data1_aligned > sigma_thresh*rms_1) & (data2_conv > sigma_thresh*rms_2)
        
        # 6. CALCULATION
        # We only want to calculate the index where the data is valid (inside the mask)
        # This prevents the "Dimension Mismatch" error.
        
        S1 = data1_aligned[mask]  # Take only valid pixels (becomes 1D array)
        S2 = data2_conv[mask]     # Take only valid pixels (becomes 1D array)
        
        v1 = map1.freq.to(u.Hz).value
        v2 = map2.freq.to(u.Hz).value
        
        alpha_map = np.full_like(map2.data, np.nan)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            # Now S1 and S2 are 1D, so alpha_vals will also be 1D
            alpha_vals = np.log10(S1 / S2) / np.log10(v1 / v2)
            
            # Now 1D fits into 1D -> No Error
            alpha_map[mask] = alpha_vals

        # 7. DISPLAY & SAVE
        st.success("Calculation Complete")
        
        # Plotting for Web
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(alpha_map, origin='lower', cmap='jet', vmin=-2, vmax=0.5)
        plt.colorbar(im, label="Alpha")
        st.pyplot(fig)

        # Prepare Download
        out_header = map2.header.copy()
        out_header['HISTORY'] = 'Convolved to common beam'
        fits.writeto('/tmp/spidx_common.fits', alpha_map, out_header, overwrite=True)
        
        with open('/tmp/spidx_common.fits', 'rb') as f:
            st.download_button("Download Result", f, "spidx_common.fits")
            
        # Cleanup
        map1.cleanup()
        map2.cleanup()

# Run the app
calculate_spectral_index_workflow()