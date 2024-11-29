# CRIMAC-dB-differencing
Development repository for dB differencing (and potentially other combinations) of multifrequency echosounder data.
Input data will be the output from the (CW) CRIMAC processing pipeline.

dB differencing
If data is collected concurrently using multiple acoustic frequencies, then operations on the data (e.g. dB differencing) is a tool to descriminate types of scatterers (e.g. fish with swimbladder and plankton).

For an example of this approach, see e.g. Ressler, P. H., P. Dalpadado, G. J. Macaulay, N. Handegard, and M. SkernMauritzen. “Acoustic Surveys of Euphausiids and Models of Baleen Whale Distribution in the Barents Sea.” Marine Ecology Progress Series 527 (May 7, 2015): 13–29. https://doi.org/10.3354/meps11257.

Typical approach:
1. Bottom detection
2. Remove samples bellow the seafloor
3. Potential noise reduction (remove spikes etc.)
4. Resample/average/smooth the two variables
5. Calculate dB difference
6. Select data that meets the specified "dB difference" criteria

