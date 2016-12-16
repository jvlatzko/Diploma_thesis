/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Basic.h
 * Implements functionality related to printing of debug/status information.
 *
 */

#ifndef H_RTF_MONITOR_H
#define H_RTF_MONITOR_H

#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <time.h>

namespace Monitor
{
    // The default monitor that is used to print debug/status info to the console. In addition
    // to a user-provided message, it also prints details about CPU usage and memory consumption.
    class DefaultMonitor
    {
    public:
        static void Display(const char* fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
            fflush(stderr);
        }
        static void Report(const char* fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            DefaultMonitor::ReportVA(fmt, args);
            va_end(args);
        }

        static void ReportVA(const char* fmt, va_list argptr)
        {
            #pragma omp critical
            {
                time_t now;
                time(&now);
                std::string timestr(ctime(&now));
                timestr.erase(timestr.end() - 1);

                fprintf(stderr, "-- %s --\n", timestr.c_str());

                vfprintf(stderr, fmt, argptr);
                fflush(stderr);
            }
        }
    };

    class NullMonitor
    {
    public:
        static void Display(const char* fmt, ...)
        {

        }
        static void Report(const char* fmt, ...)
        {

        }

        static void ReportVA(const char* fmt, va_list argptr)
        {

        }
    };
}

#endif // H_RTF_MONITOR_H
