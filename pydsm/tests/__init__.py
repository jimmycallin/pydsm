# -*- coding: utf-8 -*-
import unittest
from pydsm.tests.test_matrix import TestMatrix
from pydsm.tests.test_similarity import TestSimilarity
from pydsm.tests.test_weighting import TestWeighting


def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestMatrix))
    suite.addTest(loader.loadTestsFromTestCase(TestWeighting))
    suite.addTest(loader.loadTestsFromTestCase(TestSimilarity))
    return suite