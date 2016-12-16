/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: MPI.h
 * Defines utility routines for MPI support.
 *
 */

#ifndef H_RTF_MPI_H
#define H_RTF_MPI_H

#include "Types.h"

#define MSMPI_NO_DEPRECATE_20 1
#ifdef USE_MPI
#include <boost/mpi.hpp>

namespace MPI
{
    boost::mpi::communicator& Communicator()
    {
        static boost::mpi::communicator comm;
        return comm;
    }

    boost::mpi::environment& Environment(int argc = 0, char** argv = NULL)
    {
        static boost::mpi::environment env(argc, argv);
        return env;
    }
}

#endif // USE_MPI

#endif // H_RTF_MPI_H
