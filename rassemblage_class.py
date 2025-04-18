#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 00:34:20 2025

@author: shakib
"""

import os
import glob
import shutil

def aggregate_by_class(src_root: str, dest_root: str):
    """
    Parcourt tous les dossiers <src_root>/<dataset>/<class>/*.png
    et copie chaque image dans <dest_root>/<class>/
    """
    # 1) découvrir dynamiquement toutes les classes
    classes = set()
    for dataset in os.listdir(src_root):
        ds_path = os.path.join(src_root, dataset)
        if not os.path.isdir(ds_path): continue
        for cls in os.listdir(ds_path):
            cls_path = os.path.join(ds_path, cls)
            if os.path.isdir(cls_path):
                classes.add(cls)

    # 2) pour chaque classe, copier tous les PNG dans dest_root/cls
    for cls in classes:
        out_dir = os.path.join(dest_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        pattern = os.path.join(src_root, '*', cls, '*.png')
        files = glob.glob(pattern)
        print(f"Found {len(files)} files for class '{cls}'")
        for src in files:
            fname = os.path.basename(src)
            dst   = os.path.join(out_dir, fname)
            # Si vous préférez déplacer à la place de copier, remplacez copy2 par move
            shutil.copy2(src, dst)
        print(f" → Copied into {out_dir}\n")

if __name__ == "__main__":
    # === À ADAPTER ===
    SRC_ROOT  = "/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/val_2/spectrograms_100x100"
    DEST_ROOT = "/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/val_2/by_class"
    # ==================

    aggregate_by_class(SRC_ROOT, DEST_ROOT)
