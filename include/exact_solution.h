#ifndef __EXACT_SOLUTION_H__
#define __EXACT_SOLUTION_H__

#include <cmath>
namespace ExactSolution
{
  double
  sech(double x)
  {
    return 1.0 / std::cosh(x);
  }


  double
  exact_solution(double x, double y)
  {
    double acegen_scratch[143];
    acegen_scratch[32] = 0.15707963267948966e1;
    return std::cos(acegen_scratch[32] * (x)) *
           std::cos(acegen_scratch[32] * (y));
  };


  double
  coefficient(double(x), double(y))
  {
    double acegen_scratch[148];
    return 1e0 +
           10e0 *
             (1e0 -
              std::tanh((2e0 + 10e0 * (1e0 + cos(0.3141592653589793e1 * (y)))) *
                        (-0.4e0 + fabs((x)))));
  };


  double
  rhs(double(x), double(y))
  {
    double acegen_scratch[143];
    acegen_scratch[4]  = (x);
    acegen_scratch[24] = -0.4e0 + std::fabs(acegen_scratch[4]);
    acegen_scratch[9]  = 0.15707963267948966e1 * acegen_scratch[4];
    acegen_scratch[17] = std::cos(acegen_scratch[9]);
    acegen_scratch[22] = 0.3141592653589793e1 * (y);
    acegen_scratch[10] = acegen_scratch[22] / 2e0;
    acegen_scratch[18] = std::cos(acegen_scratch[10]);
    acegen_scratch[6]  = 2e0 * (6e0 + 5e0 * std::cos(acegen_scratch[22]));
    acegen_scratch[15] = acegen_scratch[24] * acegen_scratch[6];
    return (-0.49348022005446796e3 * acegen_scratch[17] * acegen_scratch[24] *
              std::sin(acegen_scratch[10]) * std::sin(acegen_scratch[22]) +
            0.15707963267948966e2 * acegen_scratch[18] * acegen_scratch[6] *
              std::abs(acegen_scratch[4]) * std::sin(acegen_scratch[9])) *
             std::pow(sech(acegen_scratch[15]), 2) +
           0.49348022005446796e2 * acegen_scratch[17] * acegen_scratch[18] *
             (-0.11e1 + std::tanh(acegen_scratch[15]));
  };
} // namespace ExactSolution
#endif // __EXACT_SOLUTION_H__