# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:53:30 2025

@author: adity
"""

import streamlit as st
import numpy as np
import warnings
import tempfile
import os
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import mad_std
from radio_beam import Beam
from reproject import reproject_interp
import matplotlib.pyplot as plt

# --- 1. SETUP ---
st.set_page_config(page_title="Spectral Index Pipeline", layout="wide")
warnings.filterwarnings('ignore')
st.title("Spectral Index Pipeline")

# --- 2. THE CLASS (Web Version) ---
class RadioMap:
    def __init__(self, uploaded_file, name="Map"):
        self.name = name
        
        # Save uploaded bytes to a temp file so Astropy can read it
        self.tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
        self.tfile.write(uploaded_file.getvalue())
        self.tfile.close()
        self.filepath = self.tfile.name
        
        self.data = None
        self.header = None
        self.wcs = None
        self.beam = None
        self.freq = None
        
        self.load_data()

    def load_data(self):
        with fits.open(self.filepath) as hdul:
            hdu = hdul[0] if len(hdul) > 0 and hdul[0].data is not None else hdul[1]
            self.header = hdu.header
            self.data = np.squeeze(hdu.data)
            self.wcs = WCS(self.header).celestial
            
        st.write(f"**Loaded {self.name}**")
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
                    st.info(f"Freq {self.name}: {self.freq.to(u.MHz):.2f}")
                    found = True
                    break
        if not found:
            # Replaces input() with a web form
            st.warning(f"[!] Frequency missing for {self.name}")
            val = st.number_input(f"Enter Frequency for {self.name} (MHz):", min_value=0.0, step=1.0, key=f"freq_{self.name}")
            if val == 0.0:
                st.error("Please enter a frequency > 0 to continue.")
                st.stop()
            self.freq = val * u.MHz

    def _get_beam(self):
        try:
            self.beam = Beam.from_fits_header(self.header)
            st.info(f"Beam {self.name}: {self.beam}")
        except:
            st.warning(f"[!] Beam info missing in {self.name}.")
            c1, c2, c3 = st.columns(3)
            bmaj = c1.number_input(f"{self.name} Maj Axis (arcsec):", min_value=0.0, key=f"bmaj_{self.name}")
            bmin = c2.number_input(f"{self.name} Min Axis (arcsec):", min_value=0.0, key=f"bmin_{self.name}")
            bpa = c3.number_input(f"{self.name} PA (deg):", value=0.0, key=f"bpa_{self.name}")
            
            if bmaj == 0.0 or bmin == 0.0:
                st.error("Please enter Beam Major/Minor axis to continue.")
                st.stop()
            self.beam = Beam(major=bmaj*u.arcsec, minor=bmin*u.arcsec, pa=bpa*u.deg)

    def _check_units(self):
        unit_str = self.header.get('BUNIT', '').lower()
        if 'mjy' in unit_str:
            st.write(f"[{self.name}] Converting mJy -> Jy")
            self.data = self.data / 1000.0
            self.header['BUNIT'] = 'Jy/beam'
        elif 'jy' in unit_str:
            pass 
        else:
            st.warning(f"[!] Unit unknown for {self.name}.")
            choice = st.radio(f"Select units for {self.name}:", ("Select...", "Yes (mJy)", "No (Jy)"), key=f"unit_{self.name}")
            if choice == "Select...":
                st.stop()
            if choice == "Yes (mJy)":
                self.data = self.data / 1000.0

    def cleanup(self):
        if os.path.exists(self.filepath):
            os.unlink(self.filepath)

# --- 3. CONVOLUTION ---
def convolve_to_common(source_map, target_beam):
    if source_map.beam == target_beam:
        return source_map.data

    st.write(f"--- Convolving {source_map.name} ---")
    try:
        kernel = target_beam.deconvolve(source_map.beam)
        kernel_pix = kernel.as_kernel(source_map.wcs.proj_plane_pixel_area()**0.5)
        
        from astropy.convolution import convolve_fft
        convolved_data = convolve_fft(source_map.data, kernel_pix, allow_huge=True)
        
        scale_factor = target_beam.sr.value / source_map.beam.sr.value
        return convolved_data * scale_factor
        
    except ValueError:
        st.error("[CRITICAL ERROR] Deconvolution failed.")
        return source_map.data

# --- 4. MAIN WORKFLOW ---
def calculate_spectral_index_workflow():
    st.sidebar.header("Upload Files")
    f1 = st.sidebar.file_uploader("Upload Low Freq Map (FITS)", type=["fits"])
    f2 = st.sidebar.file_uploader("Upload High Freq Map (FITS)", type=["fits"])

    if not f1 or not f2:
        st.info("Waiting for both FITS files...")
        st.stop()

    map1 = RadioMap(f1, "Map 1")
    map2 = RadioMap(f2, "Map 2")

    st.divider()
    
    # Define Common Beam
    max_axis = max(map1.beam.major, map1.beam.minor, map2.beam.major, map2.beam.minor) * 1.01 
    common_beam = Beam(major=max_axis, minor=max_axis, pa=0*u.deg)
    st.success(f"Common Beam: {common_beam.major.to(u.arcsec):.2f}")

    # Process
    with st.spinner("Processing..."):
        d1_conv = convolve_to_common(map1, common_beam)
        d2_conv = convolve_to_common(map2, common_beam)

        st.write("Regridding...")
        d1_aligned, _ = reproject_interp((d1_conv, map1.wcs), map2.wcs, shape_out=map2.data.shape, order=3)

    # Calculation
    rms1 = mad_std(d1_aligned, ignore_nan=True)
    rms2 = mad_std(d2_conv, ignore_nan=True)
    
    sigma = st.number_input("Sigma Threshold", min_value=1.0, value=3.0, step=0.5)
    
    if st.button("Calculate Map"):
        mask = (d1_aligned > sigma*rms1) & (d2_conv > sigma*rms2)
        
        S1 = d1_aligned[mask]
        S2 = d2_conv[mask]
        v1 = map1.freq.to(u.Hz).value
        v2 = map2.freq.to(u.Hz).value
        
        alpha_map = np.full_like(map2.data, np.nan)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            alpha_vals = np.log10(S1 / S2) / np.log10(v1 / v2)
            alpha_map[mask] = alpha_vals

        # Display
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(alpha_map, origin='lower', cmap='jet', vmin=-2.0, vmax=0.5)
        plt.colorbar(im, label="Spectral Index")
        st.pyplot(fig)
        
        # Download
        out_header = map2.header.copy()
        out_header['HISTORY'] = 'Spectral Index Map'
        fits.writeto("/tmp/alpha.fits", alpha_map, out_header, overwrite=True)
        
        with open("/tmp/alpha.fits", "rb") as f:
            st.download_button("Download FITS", f, "spectral_index.fits")
            
    map1.cleanup()
    map2.cleanup()

if __name__ == "__main__":
    calculate_spectral_index_workflow()