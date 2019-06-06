
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