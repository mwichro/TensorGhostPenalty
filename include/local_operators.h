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
  class Laplace
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


    Laplace()
      : cell_side_length(std::numeric_limits<double>::quiet_NaN())
      , penalty_parameter(5)
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



    template <class CutIntegratorsType>
    void
    integrate_cut_cell(CutIntegratorsType &cut_cell_integrators) const;

    void
    initialize(const std::shared_ptr<const MatrixFree<dim, Number>> &data,
               double penalty_parameter_)
    {
      penalty_parameter = penalty_parameter_;
      if (data->get_mg_level() == numbers::invalid_unsigned_int)
        this->is_level_operator = false;

      if (data->n_cell_batches() == 0)
        return;
      auto cell              = data->get_cell_iterator(0, 0);
      this->cell_side_length = cell->minimum_vertex_distance();
    }

    void
    initialize(const std::shared_ptr<const MatrixFree<dim, Number>> &data,
               const std::vector<FullMatrix<double>> &intersected_matrices)
    {
      const unsigned int cut_cell_category = 1;
      unsigned int       n_cut_cells       = 0;
      const unsigned int n_dofs            = std::pow(degree_u + 1, dim);

      data_tmp = data;

      intersected_mats = &intersected_matrices;

      FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> dummy_data(
        *data);
      const std::vector<unsigned int> dof_numbering =
        dummy_data.get_internal_dof_numbering();

      // for (auto &num : dof_numbering)
      //   std::cout << num << std::endl;


      for (unsigned int cell_b = 0; cell_b < data->n_cell_batches(); ++cell_b)
        if (data->get_cell_category(cell_b) == cut_cell_category)
          ++n_cut_cells;

      cut_matrices.reinit(n_cut_cells, n_dofs, n_dofs);
      raw_matrices.reinit(n_cut_cells, n_dofs, n_dofs);
      data_offsets = std::vector<unsigned int>(data->n_cell_batches(),
                                               numbers::invalid_unsigned_int);

      unsigned int current_offset = 0;

      for (unsigned int cell_b = 0; cell_b < data->n_cell_batches(); ++cell_b)
        {
          if (data->get_cell_category(cell_b) != cut_cell_category)
            continue;


          for (unsigned int lane = 0;
               lane < data->n_active_entries_per_cell_batch(cell_b);
               ++lane)
            {
              auto cell = data->get_cell_iterator(cell_b, lane);

              this->cell_side_length = cell->minimum_vertex_distance();

              AssertDimension(intersected_matrices[cell->user_index()].m(),
                              n_dofs);
              for (unsigned int i = 0; i < n_dofs; ++i)
                for (unsigned int j = 0; j < n_dofs; ++j)
                  {
                    cut_matrices[current_offset][i][j][lane] =
                      intersected_matrices[cell->user_index()](
                        dof_numbering[i], dof_numbering[j]);

                    raw_matrices[current_offset][i][j][lane] =
                      intersected_matrices[cell->user_index()](i, j);
                  }
            }
          data_offsets[cell_b] = current_offset;
          ++current_offset;
        }
      AssertDimension(current_offset, n_cut_cells);
    }

  private:
    std::vector<unsigned int>         data_offsets;
    Table<3, VectorizedArray<Number>> cut_matrices;
    Table<3, VectorizedArray<Number>> raw_matrices;

    const std::vector<FullMatrix<double>> *        intersected_mats;
    std::shared_ptr<const MatrixFree<dim, Number>> data_tmp;

    double cell_side_length;
    bool   is_level_operator;
    double penalty_parameter;
  };

  template <int dim, int degree_u, typename Number, int n_quadrature>
  void
  Laplace<dim, degree_u, Number, n_quadrature>::integrate_cell(
    FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> &dst,
    const unsigned int &                                            cell) const
  {
    (void)cell;
    dst.evaluate(EvaluationFlags::gradients);
    for (unsigned int q = 0; q < dst.n_q_points; ++q)
      dst.submit_gradient(dst.get_gradient(q), q);
    dst.integrate(EvaluationFlags::gradients);
  }

  template <int dim, int degree_u, typename Number, int n_quadrature>
  void
  Laplace<dim, degree_u, Number, n_quadrature>::integrate_cut_cell(
    FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> &      dst,
    const FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> &src,
    const unsigned int &cell) const
  {
    const unsigned int n_dofs = src.dofs_per_cell;

    const unsigned int offset = data_offsets[cell];
    Assert(offset != numbers::invalid_unsigned_int,
           ExcMessage("Invalid data offset"));


    AssertDimension(n_dofs, std::pow(degree_u + 1, dim));
    AlignedVector<VectorizedArray<Number>> result(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
      {
        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            result[i] += cut_matrices[offset][i][j] * src.get_dof_value(j);
          }
      }
    for (unsigned int i = 0; i < n_dofs; ++i)
      dst.submit_dof_value(result[i], i);


    const std::vector<unsigned int> dof_numbering =
      dst.get_internal_dof_numbering();
    for (unsigned int lane = 0;
         lane < this->data_tmp->n_active_entries_per_cell_batch(cell);
         ++lane)
      {
        auto         cell_dof = data_tmp->get_cell_iterator(cell, lane);
        const size_t index    = cell_dof->user_index();
        const FullMatrix<double> matrix = (*intersected_mats)[index];

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            if (matrix[dof_numbering[i]][dof_numbering[j]] !=
                cut_matrices[offset][i][j][lane])
              std::cout << "PROBLEM!!" << std::endl;
      }
  }



  template <int dim, int degree_u, typename Number, int n_quadrature>
  template <class CutIntegratorsType>
  void
  Laplace<dim, degree_u, Number, n_quadrature>::integrate_cut_cell(
    CutIntegratorsType &cut_cell_integrators) const
  {
    auto &phi_cut_cell      = cut_cell_integrators.phi_inside;
    auto &phi_cut_surface   = cut_cell_integrators.phi_surface;
    auto &src_dst_lane_view = cut_cell_integrators.src_dst_lane;

    // const unsigned int    n_dofs = src_dst_lane_view.size();
    // AlignedVector<Number> result(n_dofs);

    // const unsigned int offset = data_offsets[cell];
    // for (unsigned int i = 0; i < n_dofs; ++i)
    //   for (unsigned int j = 0; j < n_dofs; ++j)
    //     result[i] += raw_matrices[offset][i][j][lane] * src_dst_lane_view[j];
    // for (unsigned int i = 0; i < n_dofs; ++i)
    //   src_dst_lane_view[i] = result[i];
    // return;


    phi_cut_cell.evaluate(src_dst_lane_view, EvaluationFlags::gradients);
    phi_cut_surface.evaluate(src_dst_lane_view,
                             EvaluationFlags::gradients |
                               EvaluationFlags::values);

    for (const unsigned int q : phi_cut_cell.quadrature_point_indices())
      phi_cut_cell.submit_gradient(phi_cut_cell.get_gradient(q), q);

    phi_cut_cell.integrate(src_dst_lane_view,
                           EvaluationFlags::gradients,
                           false);


    const double nitsche_parameter =
      penalty_parameter * (degree_u + 1) * degree_u;
    for (const unsigned int q : phi_cut_surface.quadrature_point_indices())
      {
        const Tensor<1, dim, Number> normal = phi_cut_surface.normal_vector(q);
        const Number                 uh_minus_nGradU =
          -phi_cut_surface.get_gradient(q) * normal +
          (nitsche_parameter / cell_side_length) * phi_cut_surface.get_value(q);

        const Tensor<1, dim, Number> minus_nU =
          -phi_cut_surface.get_value(q) * normal;

        phi_cut_surface.submit_gradient(minus_nU, q);
        phi_cut_surface.submit_value(uh_minus_nGradU, q);
      }


    phi_cut_surface.integrate(src_dst_lane_view,
                              EvaluationFlags::gradients |
                                EvaluationFlags::values,
                              true);
  }



} // namespace LocalOperators
#endif
