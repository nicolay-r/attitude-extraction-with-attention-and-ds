#!/usr/bin/env bash
pushd .
cd ctx
python -u att_cnn.py > __att-cnn.txt
python -u att_ef_cnn.py > __att-frames-cnn.txt
python -u att_ef_pcnn.py > __att-frames-pcnn.txt
python -u att_frames_cnn.py > __att-frames-cnn.txt
python -u att_frames_pcnn.py > __att-frames-pcnn.txt
python -u att_pcnn.py > __att-pcnn.txt
python -u cnn.py > __cnn-log.txt
python -u pcnn.py > __pcnn-log.txt
popd

pushd .
cd mi
python -u att_cnn.py > __att-cnn.txt
python -u att_frames_cnn.py > __att-frames-cnn.txt
python -u att_frames_pcnn.py > __att-frames-pcnn.txt
python -u att_pcnn.py > __att-pcnn.txt
python -u cnn.py > __cnn-log.txt
python -u pcnn.py > __pcnn-log.txt
popd

pushd .
cd mi_att
python -u att_cnn.py > __att-cnn.txt
python -u att_frames_cnn.py > __att-frames-cnn.txt
python -u att_frames_pcnn.py > __att-frames-pcnn.txt
python -u att_pcnn.py > __att-pcnn.txt
python -u cnn.py > __cnn-log.txt
python -u pcnn.py > __pcnn-log.txt
popd
