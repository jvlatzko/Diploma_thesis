/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: QiDAGM12Filterbank.h
 * Implements a particular filterbank for use in natural image denoising.
 *
 */

#ifndef H_QIDAGM_FILTERBANK_H
#define H_QIDAGM_FILTERBANK_H

// Filterbank described in the paper:
// Qi Gao and Stefan Roth. How well do filter-based MRFs model natural images?
// In Proc. of the 34th DAGM-Symposium, 2012. Oral presentation. DAGM Prize.

class QiDAGM12Filterbank
{
public:
    static const int filter_count = 16;
    static const int filter_size_y;
    static const int filter_size_x;

    static const double filter_values[16][5][5];
};

#endif // H_QIDAGM_FILTERBANK_H
