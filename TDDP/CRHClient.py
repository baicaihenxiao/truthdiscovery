import numpy as np
import re

from truthdiscovery import MatrixDataset, DistanceMeasures, ConvergenceIterator
from truthdiscovery.algorithm import CRH
from parse_param import ParseParam

# mydata = MatrixDataset(ma.masked_values([
#     [4, 7, 0],
#     [0, 7, 8],
#     [3, 0, 5],
#     [3, 6, 8]
# ], 0))

mydata = MatrixDataset(np.array([
    [4, 7, 0],
    [4, 7, 0],
    [10, 7, 8],
    [3, 0, 5],
    [3, 6, 8]
]))

parseParam = ParseParam()
# params = parseParam.get_param_dict('iterator=l1-convergence-0.0010000000-limit-100')
# print(params)
# crh = CRH(**params)
# results = crh.run(mydata)
# print(results)
# pres = parseParam.get_output_obj(results)
# print(pres)


conIter = ConvergenceIterator(DistanceMeasures.L1, 0.001, limit=100)
crh = CRH(conIter)
results = crh.run(mydata)
print(results)
pres = parseParam.get_output_obj(results)
print(pres)


