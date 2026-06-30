# Broadband FMR Post-Processing GUI

A PyQt5 desktop application for post-processing broadband ferromagnetic
resonance (FMR) measurement data, where each file in a dataset directory
corresponds to one microwave frequency and contains a magnetic-field sweep.

## Install

```bash
pip install -r requirements.txt
```

(`openpyxl` is included for `.xlsx` slice export in Tab 4; everything else
is required.)

## Run

```bash
python main.py
```

## Workflow

1. **Data Import** — Add one or more experimental directories (each becomes
   a row you can label, e.g. "Sample A"). Optionally select a background
   directory and enable background subtraction. Click **Load / Reload All**.
   The Data Summary box shows file counts and field/frequency ranges per
   dataset.

2. **Processing Settings** — Choose background subtraction method (direct /
   normalized / division), toggle DC offset removal, detrending, and the
   Savitzky-Golay filter (window length must be odd and greater than the
   polynomial order — the app validates this live). Optionally apply
   exponential / logarithmic / gamma enhancement for visualization, and pick
   the output quantity (raw signal, absorption via cumulative integration,
   or first/second derivative). **Preview Processing** shows a quick look
   on one dataset/frequency; **Process All Datasets** runs the full pipeline
   over everything loaded in Tab 1.

3. **Heatmap Visualization** — Pick a processed dataset from the dropdown
   and view its 2D field-vs-frequency intensity map. Choose colormap,
   manual or auto color scaling, and interpolation. Zoom/pan/save/reset are
   available from the Matplotlib toolbar above the plot; a live coordinate
   readout sits underneath.

4. **Heatmap Sections** — Two sub-tabs:
   - **Frequency Slice**: pick a frequency, check which datasets to
     include, click Extract to overlay Signal vs Magnetic Field curves
     (legend uses your dataset labels), then export the slice.
   - **Field Slice**: same idea, but pick a magnetic field value and plot
     Signal vs Frequency.

5. **Peak Analysis** — Check which processed datasets to include, set
   **Number of Peaks** (the strongest N local maxima per frequency's field
   cross-section), and click **Find Peaks**. Each peak "track" (the 1st,
   2nd, ... peak, ordered by field position so it follows the same
   resonance branch across frequency rather than jumping by amplitude
   rank) is plotted as a dash-dot line of Peak Field Position (x-axis) vs
   Frequency (y-axis), one color per dataset. Frequencies with fewer
   peaks than requested are left blank (with a warning) rather than
   crashing.

   **Peak Gap Filtering**: for each peak beyond the first, a "Peak *n*
   Max Gap (T)" setting appears (so Number of Peaks = 3 shows 2 gap
   settings), defaulting to 0.05 T. If a found peak sits farther from
   Peak 1 than its allowed gap, it's treated as a noise peak: instead of
   being plotted as a stray far-field point, its position is overlapped
   onto Peak 1's, so the track stays continuous instead of jumping out to
   the noise. A summary warning reports how many points were filtered
   this way.

   **Export Peak Data...** writes a long-format table (dataset, peak
   index, frequency, field, intensity) to CSV, TXT, or Excel, reflecting
   any gap-filtered overlaps.

6. **Data Export** — Export a processed dataset's full field/frequency/
   intensity matrix (CSV, TXT, NPY, MAT) or re-render and save its heatmap
   figure (PNG, SVG, PDF, EPS) at 300/600/1200 DPI.

## File format expectations

Each FMR file should have a `field` column and a `voltageX` column (and
ideally `voltageY`); the frequency is parsed from the filename (e.g.
`FMR_mmWave5.0GHz.csv` -> 5.0 GHz). Missing/unparseable files are skipped
with a warning rather than crashing the app - check the Data Summary box
or the warning dialogs after loading/processing.

## Project layout

```
main.py
gui/
    main_window.py       Top-level window, assembles all 6 tabs
    app_state.py          Shared state object passed between tabs
    import_tab.py         Tab 1
    processing_tab.py     Tab 2
    heatmap_tab.py        Tab 3
    section_tab.py        Tab 4 (Frequency Slice + Field Slice sub-tabs)
    peak_tab.py           Tab 5 (Peak Analysis)
    export_tab.py         Tab 6
processing/
    loader.py             Directory scanning, file reading, frequency parsing
    preprocessing.py      DC removal, detrend, Savitzky-Golay, background sub,
                           enhancement, absorption/derivative computation
    dataset_processor.py  Runs preprocessing over a full dataset, builds the
                           2D matrix, frequency/field slicing
    peak_analysis.py      Per-frequency peak detection and peak-track building
plotting/
    mpl_canvas.py          Reusable embedded Matplotlib widget (toolbar +
                            crosshair readout)
utils/
    exporters.py           CSV/TXT/NPY/MAT heatmap export, slice export,
                            peak-data export, figure export
```
