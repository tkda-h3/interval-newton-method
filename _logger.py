# -*- coding: utf-8 -*-
#!/usr/bin/env python

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s][%(name)s][f: %(funcName)s line %(lineno)s]%(message)s')
fh = logging.FileHandler('find_all_solution.out', 'a')
fh.setFormatter(formatter)
logger.addHandler(fh)
