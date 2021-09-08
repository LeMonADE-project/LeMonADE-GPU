/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo--GPU--oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADEGPU.

LeMonADE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LeMonADE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------*/
template <typename Criterion >
class LocalMetropolis
{
public: 
  LocalMetropolis():LocalMetropolisOn(false),criterion() {}
  LocalMetropolis(bool LocalMetropolisOn_):LocalMetropolisOn(LocalMetropolisOn_) {}
  
  inline void setLocalMetropolisOn(bool LocalMetropolisOn_){LocalMetropolisOn=LocalMetropolisOn_;}

  inline __host__ __device__ double operator() () {return 0; }

  template <class T>
  inline __host__ __device__ double operator() (T input)
  {
    switch (LocalMetropolisOn)
    {
      case 0: return 1;
      case 1: return criterion(input);
    }
  }
  template <class T1, class T2>
  inline __host__ __device__ double operator() (T1 input1, T2 input2)
  {
    switch (LocalMetropolisOn)
    {
      case 0: return 1;
      case 1: return criterion(input1, input2);
    };
  }
private:
  bool LocalMetropolisOn;  
  Criterion criterion;
};