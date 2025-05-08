import fastremap
import numpy as np


def arrays_equal_up_to_id_ordering(arr1, arr2):
    test_uniques = fastremap.unique(arr1)
    gt_uniques = fastremap.unique(arr2)
    relabeling_dict = {}

    correct_uniques = True
    for test_unique in test_uniques:
        z, y, x = np.nonzero(arr1 == test_unique)
        cooresponding_gt_uniques = arr2[z, y, x]
        if len(fastremap.unique(cooresponding_gt_uniques)) > 1:
            correct_uniques = False
            break
        relabeling_dict[test_unique] = arr2[z[0], y[0], x[0]]

    fastremap.remap(arr1, relabeling_dict, preserve_missing_labels=True, in_place=True)
    return (
        correct_uniques
        and np.array_equal(arr1, arr2)
        and max(gt_uniques) == max(test_uniques)
    )
