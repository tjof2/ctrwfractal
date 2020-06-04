/***************************************************************************

  Copyright 2016-2020 Tom Furnival

  This file is part of ctrwfractal.

  ctrwfractal is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ctrwfractal is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ctrwfractal.  If not, see <http://www.gnu.org/licenses/>.

***************************************************************************/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <utility>
#include <armadillo>

inline double SquaredDist(const double &x1, const double &x2, const double &y1, const double &y2)
{
    double a = (x1 - x2);
    double b = (y1 - y2);
    return a * a + b * b;
}

template <typename Arg, typename... Args>
void Print(std::ostream &out, Arg &&arg, Args &&... args)
{
    out << std::forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(out << std::forward<Args>(args)), 0)...};
}

template <typename Arg, typename... Args>
void PrintFixed(const uint32_t precision, Arg &&arg, Args &&... args)
{
    Print(std::cout, std::fixed, std::setprecision(precision), arg, args...);
}

std::chrono::high_resolution_clock::time_point GetTime()
{
    return std::chrono::high_resolution_clock::now();
}

double ElapsedSeconds(std::chrono::high_resolution_clock::time_point t0,
                      std::chrono::high_resolution_clock::time_point t1)
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 1E-6);
}

template <typename Function, typename Integer_Type>
void parallel(Function const &func,
              Integer_Type dimFirst,
              Integer_Type dimLast,
              int nJobs = -1,
              uint32_t threshold = 1)
{
    uint32_t const totalCores = (nJobs > 0) ? nJobs : std::thread::hardware_concurrency();

    if ((nJobs == 0) || (totalCores <= 1) || ((dimLast - dimFirst) <= threshold)) // No parallelization or small jobs
    {
        for (auto a = dimFirst; a != dimLast; ++a)
        {
            func(a);
        }
        return;
    }
    else // std::thread parallelization
    {
        std::vector<std::thread> threads;
        if (dimLast - dimFirst <= totalCores) // case of small job numbers
        {
            for (auto index = dimFirst; index != dimLast; ++index)
                threads.emplace_back(std::thread{[&func, index]() { func(index); }});
            for (auto &th : threads)
            {
                th.join();
            }
            return;
        }

        auto const &jobSlice = [&func](Integer_Type a, Integer_Type b) { // case of more jobs than CPU cores
            if (a >= b)
            {
                return;
            }
            while (a != b)
            {
                func(a++);
            }
        };

        threads.reserve(totalCores - 1);
        uint64_t tasksPerThread = (dimLast - dimFirst + totalCores - 1) / totalCores;

        for (auto index = 0UL; index != totalCores - 1; ++index)
        {
            Integer_Type first = tasksPerThread * index + dimFirst;
            first = std::min(first, dimLast);
            Integer_Type last = first + tasksPerThread;
            last = std::min(last, dimLast);
            threads.emplace_back(std::thread{jobSlice, first, last});
        }

        jobSlice(tasksPerThread * (totalCores - 1), dimLast);
        for (auto &th : threads)
        {
            th.join();
        }
    }
};

template <typename T>
void SetMemState(T &t, int state)
{
    const_cast<arma::uhword &>(t.mem_state) = state;
}

template <typename T>
size_t GetMemState(T &t)
{
    if (t.mem && t.n_elem <= arma::arma_config::mat_prealloc)
        return 0;

    return (size_t)t.mem_state;
}

template <typename T>
inline typename T::elem_type *GetMemory(T &m)
{
    if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
    {
        typename T::elem_type *mem = arma::memory::acquire<typename T::elem_type>(m.n_elem);
        arma::arrayops::copy(mem, m.memptr(), m.n_elem);
        return mem;
    }
    else
    {
        return m.memptr();
    }
}

#endif