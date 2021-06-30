#!/usr/bin/env python3

import importlib.util
import sys

spec = importlib.util.spec_from_file_location('data', 'net/data.py')
module = importlib.util.module_from_spec(spec)
sys.modules['data'] = module
spec.loader.exec_module(module)

if __name__ == '__main__':
    data = module.TextData()
