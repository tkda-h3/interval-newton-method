# -*- coding: utf-8 -*-
#!/usr/bin/env python

import logging
import os

logfile_path = 'find_all_solution.out'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s][%(name)s][f: %(funcName)s line %(lineno)s]%(message)s')
fh = logging.FileHandler(logfile_path, 'a')
fh.setFormatter(formatter)
logger.addHandler(fh)
