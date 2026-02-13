#!/usr/bin/env python3
import argparse
import os
import yaml
from scripts.utils.logger import get_logger
from cross_dataset_ModelNet40_ScanObjectNN.scripts.build_modelnet import walk_and_process
from scripts.build_scanobjectnn import process_h5_file
import glob

def main(config_path, sessions_json, out_root):
    cfg = yaml.safe_load(open(config_path, 'r'))
    logger = get_logger('build_all', log_file=os.path.join(out_root, 'build_all.log'))
    logger.info('Starting ModelNet processing...')
    walk_and_process(cfg['modelnet']['root'], out_root, cfg['modelnet'].get('sample_surface_n',10000), cfg['parallel']['workers'], cfg['parallel']['chunk_size'])
    logger.info('Starting ScanObjectNN processing...')
    pattern = os.path.join(cfg['scanobjectnn']['root'], cfg['scanobjectnn'].get('variant','main_split_nobg'), '*.h5')
    h5_list = glob.glob(pattern)
    for h5p in h5_list:
        split_name = os.path.splitext(os.path.basename(h5p))[0]
        split = 'train' if 'train' in split_name else 'test'
        process_h5_file(h5p, out_root, split, cfg['parallel']['workers'], cfg['parallel']['chunk_size'])
    logger.info('All done.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--sessions', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    main(args.config, args.sessions, args.out)
