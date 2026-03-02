/*
 * matrix_free_operator.h
 *
 *  Created on: Sep 24, 2022
 *      Author: mwichro
 */

#ifndef INCLUDE_LOCAL_OPERATOR_H_
#define INCLUDE_LOCAL_OPERATOR_H_


#include <deal.II/base/smartpointer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <complex>
#include <fstream>
#include <iostream>
#include <tuple>

namespace LocalOperators
{
  using namespace dealii;



  using namespace dealii;
  template <int dim,
            int degree_u,
            typename Number,
            int n_quadrature = degree_u + 1>
  class LaplaceSBM
  {
  public:
    const constexpr static int  dimension    = dim;
    const constexpr static int  fe_degree    = degree_u;
    const constexpr static int  n_q_points   = n_quadrature;
    const constexpr static int  n_components = 1;
    const constexpr static bool is_vector    = false;

    typedef VectorizedArray<Number>    VectorizedNumber;
    const constexpr static std::size_t n_lanes = VectorizedNumber::size();
    using value_type                           = Number;

    const constexpr static EvaluationFlags::EvaluationFlags evaluation_flags =
      EvaluationFlags::gradients;
    const constexpr static EvaluationFlags::EvaluationFlags integration_flags =
      EvaluationFlags::gradients;


    LaplaceSBM()
    {}

    void
    integrate_cell(
      FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> &dst,
      const unsigned int &cell) const;

    void
    integrate_cut_cell(
      FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> &      dst,
      const FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> &src,
      const unsigned int &cell) const;


    void
    initialize(const std::shared_ptr<const MatrixFree<dim, Number>> &data)
    {}

  private:
  };

} // namespace LocalOperators
#endif
