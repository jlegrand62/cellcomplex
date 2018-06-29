# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015-2016 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenaleaLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np
from scipy import ndimage as nd


def array_unique(array, return_index=False):
    """
    Return an array made of the unique occurrence of each rows.

    Parameters
    ----------
    array : np.array
        the array to compare by rows
    return_index : bool, optional
        if False (default) do NOT return the index of the unique rows, else do
        return them

    Returns
    -------
    array_unique : np.array
        the array made of unique rows
    unique_rows : np.array, if return_index == True
        index of the unique rows

    Examples
    --------
    >>> from openalea.cellcomplex.property_topomesh.utils.array_tools import array_unique
    >>> a = np.array([[0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5]])
    >>> array_unique(a)
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5]])
    """
    _, unique_rows = np.unique(np.ascontiguousarray(array).view(
        np.dtype((np.void, array.dtype.itemsize * array.shape[1]))),
                               return_index=True)
    if return_index:
        return array[unique_rows], unique_rows
    else:
        return array[unique_rows]


def where_list(array, values):
    """
    Search list of 'values' in 'array', and return their index within the array.

    Parameters
    ----------
    array : np.array
        array to be searched
    values : list
        list of values to search in the array

    Returns
    -------
    where_list : list
        list of indexes corresponding to given 'values', number of arrays within
        this list depend on the dimensionality of the given array

    Examples
    --------
    >>> from openalea.cellcomplex.property_topomesh.utils.array_tools import where_list
    >>> a = np.array([[0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5]])
    >>> l = [0, 5]
    >>> where_list(a, l)
    (array([0, 1, 2, 3]), array([0, 0, 4, 4]))
    """
    mask = nd.sum(np.ones_like(values), values, index=array)
    return np.where(mask > 0)


def array_difference(array, subarray):
    """
    Compute the difference, in terms of elements, between two arrays 'array' &
    'subarray'.
    The returned array only contains elements of 'array' missing from 'subarray'.

    Parameters
    ----------
    array : np.array
        array containing the elements to be removed if found in 'subarray'
    subarray : np.array
        array containing the elements to remove from 'array'

    Returns
    -------
    array_difference : np.array
        the numpy array made of 'array' missing in 'subarray'.

    Examples
    --------
    >>> from openalea.cellcomplex.property_topomesh.utils.array_tools import array_difference
    >>> a = np.array([[0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5]])
    >>> b = np.array([0, 1, 2, 3, 4])
    >>> array_difference(a, b)
    """
    # TODO: set(np.array) does not work: "TypeError: unhashable type: 'numpy.ndarray'" !!!!
    # TODO: fix and add the result of 'array_difference(a, b)' in the example
    import numpy as np
    return np.array(list(set(array).difference(set(subarray))))


def weighted_percentile(values, percentiles, sample_weight=None,
                        values_sorted=False):
    """

    Parameters
    ----------
    values
    percentiles
    sample_weight
    values_sorted

    Returns
    -------

    """
    values = np.array(values)
    percentiles = np.array(percentiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(percentiles >= 0) and np.all(
        percentiles <= 100), 'percentiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_percentiles = 100. * (
                np.cumsum(sample_weight) - 0.5 * sample_weight) / np.sum(
        sample_weight)
    return np.interp(percentiles, weighted_percentiles, values)
