import os
import yaml
import shutil
import argparse

from src.bo import BO
from src.utils import Logger


# Main =======================================================================
def main():

    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.yaml', help='Path to the config file')
    parser.add_argument('-i', '--iseed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('-r', '--rseed', type=int, default=0, help='Random seed for running')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['result_dir'] = os.path.join(cfg['result_dir'], f'iseed{args.iseed}_rseed{args.rseed}/')
    if not os.path.exists(cfg['result_dir']):
        os.makedirs(cfg['result_dir'])
    shutil.copy(args.config, os.path.join(cfg['result_dir'], 'config.yaml'))
    cfg['iseed'] = args.iseed
    cfg['rseed'] = args.rseed
    # Logger
    logger = Logger(cfg['result_dir'])
    cfg['logger'] = logger

    # BO START
    message = f'BO START: {cfg["name"]}, iseed={args.iseed}, rseed={args.rseed}\n'
    ascii_art = r'    ____  ____     ______________    ____  ______' + '\n' + \
                r'   / __ )/ __ \   / ___/_  __/   |  / __ \/_  __/' + '\n' + \
                r'  / __  / / / /   \__ \ / / / /| | / /_/ / / /   ' + '\n' + \
                r' / /_/ / /_/ /   ___/ // / / ___ |/ _, _/ / /    ' + '\n' + \
                r'/_____/\____/   /____//_/ /_/  |_/_/ |_| /_/     ' + '\n'
    logger.info(message + ascii_art)

    # Bayesian Optimization
    bo = BO(**cfg)
    bo.initialize()
    bo.run()
    bo.save()

    # BO END
    message = f'BO END: {cfg["name"]}, iseed={args.iseed}, rseed={args.rseed}\n'
    ascii_art = r'    ____  ____     _______   ______  ' + '\n' + \
                r'   / __ )/ __ \   / ____/ | / / __ \ ' + '\n' + \
                r'  / __  / / / /  / __/ /  |/ / / / / ' + '\n' + \
                r' / /_/ / /_/ /  / /___/ /|  / /_/ /  ' + '\n' + \
                r'/_____/\____/  /_____/_/ |_/_____/   ' + '\n'
    logger.info(message + ascii_art)


if __name__ == '__main__':
    main()