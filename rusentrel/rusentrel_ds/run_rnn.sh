#!/usr/bin/env bash
pushd .
cd ctx
python -u att_bilstm.py > __att-bilstm.txt
python -u att_frames_bilstm.py > __att-frames-bilstm.txt
python -u att_hidden_z_yang.py > __att-hidden-zyang.txt
python -u bilstm.py > __bilstm-log.txt
python -u ian_ef.py > __ian-ends.txt
python -u ian_ends.py > __ian-ends.txt
python -u ian_frames.py > __ian-frames.txt
python -u lstm.py > __lstm-log.txt
python -u rcnn.py > __rcnn-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
popd

pushd .
cd mi
python -u att_bilstm.py > __att-bilstm.txt
python -u att_frames_bilstm.py > __att-frames-bilstm.txt
python -u att_hidden_z_yang.py > __att-hidden-zyang.txt
python -u bilstm.py > __bilstm-log.txt
python -u ian_ends.py > __ian-ends.txt
python -u ian_frames.py > __ian-frames.txt
python -u lstm.py > __lstm-log.txt
python -u rcnn.py > __rcnn-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
popd

pushd .
cd mi_att
python -u att_bilstm.py > __att-bilstm.txt
python -u att_frames_bilstm.py > __att-frames-bilstm.txt
python -u att_hidden_z_yang.py > __att-hidden-zyang.txt
python -u bilstm.py > __bilstm-log.txt
python -u ian_ends.py > __ian-ends.txt
python -u ian_frames.py > __ian-frames.txt
python -u lstm.py > __lstm-log.txt
python -u rcnn.py > __rcnn-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
popd

