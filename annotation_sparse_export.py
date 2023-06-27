import argparse
import os
import time
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from annotation.core.ArchHandler import ArchHandler
import sys
import warnings

import numpy as np
import os

we_already_have = ['P1', 'P10', 'P100', 'P101', 'P102', 'P103', 'P104', 'P105', 'P106', 'P107', 'P108', 'P109', 'P11', 'P110', 'P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P12', 'P120', 'P121', 'P122', 'P123', 'P124', 'P125', 'P126', 'P127', 'P128', 'P129', 'P13', 'P130', 'P131', 'P132', 'P133', 'P134', 'P135', 'P136', 'P137', 'P138', 'P139', 'P14', 'P140', 'P141', 'P142', 'P143', 'P144', 'P145', 'P146', 'P147', 'P148', 'P149', 'P15', 'P150', 'P151', 'P152', 'P16', 'P17', 'P18', 'P188', 'P19', 'P190', 'P191', 'P192', 'P193', 'P194', 'P195', 'P196', 'P197', 'P198', 'P199', 'P2', 'P20', 'P200', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'P207', 'P208', 'P209', 'P20900', 'P21', 'P210', 'P211', 'P212', 'P213', 'P214', 'P215', 'P216', 'P217', 'P218', 'P219', 'P22', 'P220', 'P221', 'P222', 'P223', 'P224', 'P225', 'P226', 'P227', 'P228', 'P229', 'P23', 'P230', 'P231', 'P232', 'P233', 'P234', 'P235', 'P236', 'P237', 'P238', 'P239', 'P24', 'P240', 'P241', 'P242', 'P243', 'P244', 'P245', 'P246', 'P247', 'P248', 'P249', 'P25', 'P250', 'P251', 'P252', 'P253', 'P254', 'P255', 'P256', 'P257', 'P258', 'P259', 'P26', 'P260', 'P261', 'P262', 'P263', 'P264', 'P27', 'P28', 'P29', 'P3', 'P30', 'P31', 'P32', 'P321', 'P322', 'P323', 'P324', 'P325', 'P326', 'P327', 'P328', 'P329', 'P33', 'P330', 'P331', 'P332', 'P333', 'P334', 'P335', 'P33500', 'P336', 'P337', 'P338', 'P339', 'P34', 'P340', 'P341', 'P342', 'P343', 'P344', 'P345', 'P346', 'P347', 'P348', 'P349', 'P35', 'P350', 'P351', 'P352', 'P353', 'P354', 'P355', 'P356', 'P357', 'P358', 'P359', 'P36', 'P360', 'P361', 'P362', 'P363', 'P364', 'P365', 'P366', 'P367', 'P368', 'P369', 'P37', 'P370', 'P371', 'P372', 'P373', 'P376', 'P377', 'P378', 'P379', 'P38', 'P380', 'P381', 'P382', 'P383', 'P384', 'P385', 'P386', 'P387', 'P388', 'P389', 'P39', 'P390', 'P391', 'P392', 'P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399', 'P4', 'P40', 'P400', 'P401', 'P402', 'P403', 'P404', 'P405', 'P406', 'P407', 'P408', 'P409', 'P41', 'P410', 'P411', 'P412', 'P413', 'P414', 'P41400', 'P415', 'P416', 'P417', 'P418', 'P419', 'P42', 'P420', 'P421', 'P422', 'P423', 'P424', 'P425', 'P426', 'P427', 'P428', 'P429', 'P430', 'P431', 'P432', 'P433', 'P434', 'P435', 'P436', 'P437', 'P438', 'P439', 'P44', 'P446', 'P45', 'P450', 'P455', 'P46', 'P460', 'P465', 'P47', 'P475', 'P48', 'P480', 'P49', 'P5', 'P50', 'P504', 'P51', 'P52', 'P520', 'P523', 'P53', 'P54', 'P55', 'P56', 'P57', 'P58', 'P59', 'P6', 'P60', 'P61', 'P62', 'P63', 'P64', 'P65', 'P66', 'P67', 'P68', 'P69', 'P7', 'P70', 'P71', 'P72', 'P73', 'P74', 'P75', 'P76', 'P77', 'P78', 'P79', 'P8', 'P80', 'P81', 'P82', 'P83', 'P84', 'P85', 'P86', 'P87', 'P88', 'P89', 'P9', 'P90', 'P91', 'P92', 'P93', 'P94', 'P95', 'P96', 'P97', 'P98', 'P99']
new_patients = []
to_skip = ['P189']
error_patients = []

output_folder = r"E:\IAN_Maxillo_dataset\dataset_for_public\DENSE"
for root, dirs, files in os.walk(r"E:\MaxilloReformed"):
        patient = os.path.basename(root)
        if patient[0] != 'P':
            continue
            
        if 'DICOMDIR' not in files:
            print("Patient {patient} is missing the DICOMDIR")
            continue
            
        if patient in we_already_have: continue
        if patient in to_skip: continue
        
        print(patient)
        dicomdir = os.path.join(root, 'DICOMDIR')
        if not os.path.exists(os.path.join(root, 'gt_sparse.npy')) or not os.path.exists(os.path.join(root, 'volume.npy')):
            try:
                ah = ArchHandler(dicomdir)
                ah.__init__(dicomdir)
                ah.compute_initial_state(96, want_side_volume=False)
                ah.extract_data_from_gt(load_annotations=False)
                #ah.compute_side_volume(ah.SIDE_VOLUME_SCALE, True)

                ah.export_sparse_volume(forced=True)
                ah.export_volume(forced=True)
                
            except Exception as e:
                print(e)
                error_patients.append(patient)
                continue
        os.makedirs(os.path.join(output_folder, patient), exist_ok=True)
        os.rename(
            os.path.join(root, 'gt_sparse.npy'),
            os.path.join(output_folder, patient, 'gt_sparse.npy')
        )
        os.rename(
            os.path.join(root, 'volume.npy'),
            os.path.join(output_folder, patient, 'volume.npy')
        )
        new_patients.append(patient)

print(f"New: {new_patients} ({len(new_patients)})")