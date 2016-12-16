/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: RFSFilterbank.h
 * Implements a particular filterbank for use in natural image denoising.
 *
 */

#ifndef H_RFS_FILTERBANK_H
#define H_RFS_FILTERBANK_H

class RFSFilterbank
{
public:
    static const int filter_count = 32;
    static const int filter_size_y;
    static const int filter_size_x;

    // http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
    static const double filter_values[32][17][17];
};

#endif // H_RFS_FILTERBANK_H
