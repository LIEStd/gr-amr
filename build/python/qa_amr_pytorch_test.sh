#!/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir="/home/lie/gnuradio/project/gr-amr/python"
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/lie/gnuradio/project/gr-amr/build/python":$PATH
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/lie/gnuradio/project/gr-amr/build/swig:$PYTHONPATH
/usr/bin/python3 /home/lie/gnuradio/project/gr-amr/python/qa_amr_pytorch.py 
