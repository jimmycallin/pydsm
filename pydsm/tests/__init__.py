# -*- coding: utf-8 -*-
import unittest
from pydsm.tests.test_matrix import TestMatrix

def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestMatrix))
    return suite