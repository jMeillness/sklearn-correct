#!/usr/bin/env python

import os
import setuptools

VERSION = '0.0.1.dev100'

INSTALL_REQUIRES = [
    'scikit-learn',
    ]

TEST_REQUIRES = [
    'coverage',
    ]

setuptools.setup(
    name = 'sklc',
    version = VERSION,
    description = 'sklearn for error correction',
    packages = ['sklc'],
    install_requires = INSTALL_REQUIRES,
    #extras_require = {
    #    'tests': TEST_REQUIRES,
    #    },
    #test_suite = 'tests',
    #test_loader = 'unittest:TestLoader'
    )
