# Opt2Q
![test build](https://github.com/LoLab-MSM/Opt2Q/actions/workflows/python-package.yml/badge.svg?branch=master)

Some model calibrations in this package require a different branch of PyDREAM to run properly. 
You can install that via pip, using the following: 
<pre>
    <code>
        $pip install git+https://github.com/LoLab-MSM/PyDREAM.git@fix_acceptance_rate_reporting 
    </code>
</pre>

In order to run plots of the calibrations you must first download the calibration results by running 
`download_calibration_results.py` in its example directory. 
