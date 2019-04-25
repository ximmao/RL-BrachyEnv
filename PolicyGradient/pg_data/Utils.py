import numpy as np
import os
from scipy import interpolate

matrix_size = (167, 167)

def read_file(directory):
    temp = []
    with open(directory, "r") as my_file:
        for line in my_file:
            temp.append(line.strip().split(","))
    return np.array(temp).astype(float)

def generate_plan_dict(prefix):

    """
    Given dwell duration list, generate corresponding plan matrix
    """

    for i in range(1, 11):
        assert os.path.exists(os.path.join(prefix, "plan_"+str(i)+".txt")), "plan_"+str(i)+".txt not found!"
    plan_dict = {}
    total_plan = np.zeros(matrix_size)
    for idx in range(1, 11):
        plan_dict["plan_"+str(idx)] = read_file(os.path.join(prefix, "plan_"+str(idx)+".txt"))
        np.count_nonzero(plan_dict["plan_"+str(idx)]) == 1
    return plan_dict

def generate_contour_fixed(prefix):
    index_per_contour = [6, 2, 5]
    assert isinstance(index_per_contour, list)
    assert len(index_per_contour) == 3

    contour = read_file(os.path.join(prefix, "ct_contour.txt"))
    num_ctv, num_oar1, num_oar2 = np.count_nonzero(contour == float(index_per_contour[0])), np.count_nonzero(contour == float(index_per_contour[1])), np.count_nonzero(contour == float(index_per_contour[2]))
    bool_ctv, bool_oar1, bool_oar2 = (contour == float(index_per_contour[0])), (contour == float(index_per_contour[1])), (contour == float(index_per_contour[2]))
    contour[bool_ctv] = 2.
    contour[bool_oar1] = 3.
    contour[bool_oar2] = 4.
    assert contour.shape == matrix_size, contour.shape
    #print(num_ctv, num_oar1, num_oar2)
    assert np.count_nonzero(contour == 2.) == num_ctv
    assert np.count_nonzero(contour == 3.) == num_oar1
    assert np.count_nonzero(contour == 4.) == num_oar2

    return contour

def D(ctr_dose, max_v, x):
    unit_volume = 3 * 3 * 3 / 1000.0
    num_bins = int(max_v*10) + 1

    hist, bin_edges = np.histogram(ctr_dose, bins=num_bins, range=(0, max_v+0.1))
    volume_hist = hist * unit_volume
    volume_hist = np.append(np.trim_zeros(volume_hist, trim="b"), 0)
    cum_dvh = np.cumsum(volume_hist[::-1])[::-1]

    cum_dvh = cum_dvh * 100.0 / cum_dvh[0]
    f = interpolate.interp1d(cum_dvh, bin_edges[:len(cum_dvh)], kind="linear")
    return f(x)

def generate_dose_dict(prefix):

    """
    generate corresponding dose matrix
    """

    for i in range(1, 11):
        assert os.path.exists(os.path.join(prefix, "dose_"+str(i)+".txt")), "dose_"+str(i)+".txt not found!"
    dose_dict = {}
    total_plan = np.zeros(matrix_size)
    for idx in range(1, 11):
        dose_dict["dose_"+str(idx)] = read_file(os.path.join(prefix, "dose_"+str(idx)+".txt"))
        assert dose_dict["dose_"+str(idx)].shape == matrix_size, dose_dict["dose_"+str(idx)].shape
    return dose_dict
