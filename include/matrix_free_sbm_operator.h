/*
 * matrix_free_operator.h
 *
 *  Created on: Sep 22, 2022
 *      Author: mwichro
 */

#ifndef INCLUDE_MATRIX_FREE_OPERATOR_H_
#define INCLUDE_MATRIX_FREE_OPERATOR_H_



#include <deal.II/base/smartpointer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/tensor_product_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <tuple>

#include "cut_cell_generator.h"

namespace SBM
{
  using namespace dealii;

  constexpr double
  get_penalty_factor(const unsigned &fe_degree)
  {
    return 5 * fe_degree * (fe_degree + 1);
  }

  class Timer
  {
  public:
    Timer(double &output)
      : output(output)
      , start(std::chrono::high_resolution_clock::now())
    {}
    ~Timer()
    {
      auto end = std::chrono::high_resolution_clock::now();
      output +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
        1000.;
    }

  private:
    double &                                       output;
    std::chrono::high_resolution_clock::time_point start;
  };



  enum CellStatus
  {
    inside     = 0,
    cut        = 1,
    outside    = 2,
    intefacial = 3

  };

  static constexpr bool
  is_cell_inside(const unsigned int &cell_cat)
  {
    return cell_cat == CellStatus::inside || cell_cat == CellStatus::intefacial;
  }
  static constexpr bool
  is_cell_outside(const unsigned int &cell_cat)
  {
    return cell_cat == CellStatus::outside || cell_cat == CellStatus::cut;
  }



  template <typename OPERATOR, typename VECTOR>
  class MatrixFreeOperator
    : public MatrixFreeOperators::Base<OPERATOR::dimension, VECTOR>
  {
  public:
    //  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
    typedef VECTOR                      VectorType;
    typedef OPERATOR                    LocalOperator;
    typedef typename VECTOR::value_type Number;
    typedef VectorizedArray<Number>     VectorizedNumber;


    const constexpr static int         dim          = OPERATOR::dimension;
    const constexpr static int         fe_degree    = OPERATOR::fe_degree;
    const constexpr static int         n_components = OPERATOR::n_components;
    const constexpr static int         n_q_points   = OPERATOR::n_q_points;
    const constexpr static bool        is_vector    = OPERATOR::is_vector;
    const constexpr static std::size_t n_lanes      = VectorizedNumber::size();

    const static constexpr EvaluationFlags::EvaluationFlags evaluation_flags =
      OPERATOR::evaluation_flags;
    const static constexpr EvaluationFlags::EvaluationFlags integration_flags =
      OPERATOR::integration_flags;

    MatrixFreeOperator()
      // : MatrixFreeOperators::Base<dim, VectorType>()
      : shifts_initialized(false)
      , faces_in_diagonal(false)
      , nitsche_parameter(get_penalty_factor(fe_degree))
      , do_interior_faces(true)
      , n_sbm_faces(0)
      , mpi_communicator(MPI_COMM_WORLD)
      , interior_time(0)
      , boundary_face_time(0)
      , interior_face_time(0)
      , mpi_time(0)
      , total_time(0)
    {
      selected_rows.resize(1);
      selected_rows[0] = 0;
      static_assert(
        std::is_same<typename LocalOperator::value_type, Number>::value);
      local_operator = std::make_shared<LocalOperator>();
    }


    void
    intialize_shifts(const VectorType &     level_set,
                     const DoFHandler<dim> &level_set_dof_handler);

    void
    output_shifts();


    template <typename MATRIX>
    void
    assemble_matrix(MATRIX &                         matrix,
                    const FiniteElement<dim> &       fe,
                    const AffineConstraints<double> &constrains) const;

    void
    assemble_rhs(VectorType &dst);

    virtual void
    apply_add(VectorType &dst, const VectorType &src) const override;

    template <class... Args>
    auto
    initialize_local_operator(Args &...args)
    {
      return local_operator->initialize(this->data, args...);
    }

    template <class... Args>
    auto
    initialize_local_operator(const Args &...args)
    {
      return local_operator->initialize(this->data, args...);
    }

    auto &
    get_local_operator()
    {
      return local_operator;
    }

    const auto &
    get_local_operator() const
    {
      return local_operator;
    }

    void
    reset_timing()
    {
      interior_time      = 0;
      boundary_face_time = 0;
      interior_face_time = 0;
      mpi_time           = 0;
      total_time         = 0;
    }

    void
    vmult(VectorType &dst, const VectorType &src) const;

    auto const
    number_sbm_faces() const
    {
      unsigned int global_sbm_faces = 0;
      MPI_Allreduce(&this->n_sbm_faces,
                    &global_sbm_faces,
                    1,
                    MPI_UNSIGNED,
                    MPI_SUM,
                    mpi_communicator);
      return global_sbm_faces;
    }

    void
    compute_diagonal() override;

    std::pair<Table<3, VectorizedNumber>, std::vector<unsigned int>>
    assemble_interface_matrices(
      const SparseMatrix<double> &reference = SparseMatrix<double>()) const;

    std::array<double, 5>
    get_timing(bool print) const;


    template <typename OtherVector>
    static std::vector<unsigned int>
    generate_category_vector(
      const DoFHandler<dim> &dof_handler,
      const OtherVector &    level_set,
      const double           lambda,
      const unsigned int     level = numbers::invalid_unsigned_int);

    std::size_t
    memory_consumption_custom() const;



  private:
    void
    local_apply(const MatrixFree<dim, Number> &              data,
                VectorType &                                 dst,
                const VectorType &                           src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    apply_face(const MatrixFree<dim, Number> &              data,
               VectorType &                                 dst,
               const VectorType &                           src,
               const std::pair<unsigned int, unsigned int> &face_range) const;

    void
    boundary_evaluation_kernel(
      FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> &face_phi,
      FEPointEvaluation<n_components, dim, dim, Number> &         shifted_phi,
      const Number &normal_sign) const;

    void
    internal_face_evaluation_kernel(
      FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> &phi_inner,
      FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> &phi_outer)
      const;


    void
    local_compute_cell_diagonal(
      const MatrixFree<dim, Number> &              data,
      VectorType &                                 dst,
      const unsigned int &                         component,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_compute_face_diagonal(
      const MatrixFree<dim, Number> &              data,
      VectorType &                                 dst,
      const unsigned int &                         component,
      const std::pair<unsigned int, unsigned int> &face_range) const;

    void
    local_compute_boundary_diagonal(
      const MatrixFree<dim, Number> &,
      VectorType &,
      const unsigned int &,
      const std::pair<unsigned int, unsigned int> &) const
    {}



    void
    apply_boundary(const MatrixFree<dim, Number> &,
                   LinearAlgebra::distributed::Vector<Number> &,
                   const LinearAlgebra::distributed::Vector<Number> &,
                   const std::pair<unsigned int, unsigned int> &) const
    {}


    unsigned int n_dofs_per_face;

    std::shared_ptr<LocalOperator> local_operator;
    std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>
      surface_mapping_info;

    bool                      shifts_initialized;
    std::vector<unsigned int> faces_offsets;
    bool                      faces_in_diagonal;
    const VectorizedNumber    nitsche_parameter;
    mutable bool              do_interior_faces;

    unsigned int n_sbm_faces;

    std::vector<unsigned int> selected_rows;

    MPI_Comm mpi_communicator;

    mutable double interior_time;
    mutable double boundary_face_time;
    mutable double interior_face_time;
    mutable double mpi_time;
    mutable double total_time;
  };



  template <typename OPERATOR, typename VECTOR>
  inline void
  MatrixFreeOperator<OPERATOR, VECTOR>::local_apply(
    const MatrixFree<dim, Number> &              data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points, 1, Number> phi_cell(data, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        const unsigned int cell_cat = data.get_cell_category(cell);
        if (is_cell_inside(cell_cat))
          {
            Timer timer(interior_time);
            phi_cell.reinit(cell);
            phi_cell.read_dof_values(src);
            phi_cell.evaluate(EvaluationFlags::gradients);

            for (const unsigned int q : phi_cell.quadrature_point_indices())
              phi_cell.submit_gradient(phi_cell.get_gradient(q), q);

            phi_cell.integrate(EvaluationFlags::gradients);
            phi_cell.distribute_local_to_global(dst);
          }
        else if (cell_cat == CellStatus::cut)
          {}
        else if (cell_cat == CellStatus::outside)
          {}
        else
          {
            // Assert(false, ExcNotImplemented());
          }
      }

    auto end_time = std::chrono::high_resolution_clock::now();
    interior_time +=
      std::chrono::duration<double>(end_time - start_time).count();
  }



  template <typename OPERATOR, typename VECTOR>
  inline void
  MatrixFreeOperator<OPERATOR, VECTOR>::boundary_evaluation_kernel(
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> &face_phi,
    FEPointEvaluation<n_components, dim, dim, Number> &         shifted_phi,
    const Number &normal_sign) const
  {
    const unsigned int face            = face_phi.get_cell_or_face_batch_id();
    const auto         offset          = faces_offsets[face];
    const unsigned int n_dofs_per_cell = face_phi.dofs_per_cell;

    auto                   cell = this->data->get_cell_iterator(0, 0);
    double                 cell_side_length = cell->minimum_vertex_distance();
    const VectorizedNumber inverse_length_normal_to_face =
      1. / cell_side_length;


    const unsigned int              n_quadrature_points = face_phi.n_q_points;
    AlignedVector<VectorizedNumber> shifted_values(n_quadrature_points);

    Assert(offset != numbers::invalid_unsigned_int, ExcInternalError());
    for (unsigned lane = 0;
         lane < this->data->n_active_entries_per_face_batch(face);
         ++lane)
      {
        shifted_phi.reinit(offset + lane);

        StridedArrayView<const Number, n_lanes> strided_src_dst_view(
          &(face_phi.begin_dof_values()[0][lane]), n_dofs_per_cell);
        shifted_phi.evaluate(strided_src_dst_view, EvaluationFlags::values);

        for (unsigned int q = 0; q < n_quadrature_points; ++q)
          {
            shifted_values[q][lane] = shifted_phi.get_value(q);
          }
      }

    face_phi.evaluate(EvaluationFlags::gradients);

    for (const unsigned int q : face_phi.quadrature_point_indices())
      {
        face_phi.submit_value(-normal_sign * face_phi.normal_vector(q) *
                                face_phi.get_gradient(q),
                              q);
        face_phi.submit_normal_derivative(normal_sign * shifted_values[q], q);
      }
    face_phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }



  template <typename OPERATOR, typename VECTOR>
  inline void
  MatrixFreeOperator<OPERATOR, VECTOR>::internal_face_evaluation_kernel(
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> &phi_inner,
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> &phi_outer) const
  {
    phi_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    const VectorizedNumber inverse_length_normal_to_face =
      0.5 * (std::abs((phi_inner.normal_vector(0) *
                       phi_inner.inverse_jacobian(0))[dim - 1]) +
             std::abs((phi_outer.normal_vector(0) *
                       phi_outer.inverse_jacobian(0))[dim - 1]));
    const VectorizedNumber sigma =
      inverse_length_normal_to_face * nitsche_parameter;

    for (const unsigned int q : phi_inner.quadrature_point_indices())
      {
        const VectorizedNumber solution_jump =
          (phi_inner.get_value(q) - phi_outer.get_value(q));
        const VectorizedNumber average_normal_derivative =
          (phi_inner.get_normal_derivative(q) +
           phi_outer.get_normal_derivative(q)) *
          VectorizedNumber(0.5);
        const VectorizedNumber test_by_value =
          solution_jump * sigma - average_normal_derivative;

        phi_inner.submit_value(test_by_value, q);
        phi_outer.submit_value(-test_by_value, q);

        phi_inner.submit_normal_derivative(-solution_jump *
                                             VectorizedNumber(0.5),
                                           q);
        phi_outer.submit_normal_derivative(-solution_jump *
                                             VectorizedNumber(0.5),
                                           q);
      }


    phi_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }



  template <typename OPERATOR, typename VECTOR>
  inline void
  MatrixFreeOperator<OPERATOR, VECTOR>::apply_face(
    const MatrixFree<dim, Number> &              data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    // auto   cell             = this->data->get_cell_iterator(0, 0);
    // double cell_side_length = cell->minimum_vertex_distance();
    // const VectorizedNumber inverse_length_normal_to_face = 1. /
    // cell_side_length;

    Assert(shifts_initialized,
           ExcMessage(
             "Shifts not initialized. Call intialize_shifts() first."));
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(data,
                                                                         true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_outer(data,
                                                                         false);


    // fixme: hacked FEPointEvaluation
    FE_DGQ<dim>                                       fe_fake(fe_degree);
    FEPointEvaluation<n_components, dim, dim, Number> shifted_phi(
      *surface_mapping_info, fe_fake);

    const unsigned int              n_quadrature_points = phi_inner.n_q_points;
    AlignedVector<VectorizedNumber> shifted_values(n_quadrature_points);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        const auto evaluate = [&](auto &phi, const auto n_sign) {
          Timer timer(boundary_face_time);
          phi.reinit(face);
          phi.read_dof_values(src);
          // ========================
          boundary_evaluation_kernel(phi, shifted_phi, n_sign);
          // ========================
          phi.distribute_local_to_global(dst);
        };

        const std::pair<unsigned int, unsigned int> cell_cat =
          data.get_face_category(face);

        // skip faces that are not cut
        if (is_cell_inside(cell_cat.first) && is_cell_outside(cell_cat.second))
          {
            evaluate(phi_inner, 1.);
          }
        else if (is_cell_outside(cell_cat.first) &&
                 is_cell_inside(cell_cat.second))
          {
            evaluate(phi_outer, -1.);
          }
        else if (is_cell_inside(cell_cat.first) &&
                 is_cell_inside(cell_cat.second))
          {
            if (!do_interior_faces)
              continue;
            Timer timer(interior_face_time);
            phi_inner.reinit(face);
            phi_outer.reinit(face);

            phi_inner.read_dof_values(src);
            phi_outer.read_dof_values(src);
            // ========================
            internal_face_evaluation_kernel(phi_inner, phi_outer);
            // ========================
            phi_inner.distribute_local_to_global(dst);
            phi_outer.distribute_local_to_global(dst);
          }
        else
          continue;
      }
  }



  template <typename OPERATOR, typename VECTOR>
  void
  MatrixFreeOperator<OPERATOR, VECTOR>::intialize_shifts(
    const VectorType &     level_set,
    const DoFHandler<dim> &level_set_dof_handler)
  {
    const auto &mapping = StaticMappingQ1<dim>::mapping;

    const unsigned int level = this->data->get_mg_level();
    n_sbm_faces              = 0;

    faces_offsets.clear();
    faces_offsets.resize(this->data->n_inner_face_batches(),
                         numbers::invalid_unsigned_int);


    FEFaceEvaluation<dim, fe_degree, n_q_points, 1, Number> phi_inner(
      *this->data, true);
    FEFaceEvaluation<dim, fe_degree, n_q_points, 1, Number> phi_outer(
      *this->data, false);

    const unsigned int n_quadrature_points = phi_inner.n_q_points;

    std::vector<Point<dim>> face_quad_points(n_quadrature_points);

    std::vector<typename Triangulation<dim, dim>::cell_iterator> cells_vector;
    std::vector<std::vector<Point<dim>>> closest_points_vector;

    // for debugging:
    std::vector<std::pair<Point<dim>, Point<dim>>> shifts;

    ClosestPoint::ShiftsGenerator<dim, VectorType> shifts_generator(
      level_set, level_set_dof_handler, level);


    unsigned int current_offset = 0;
    for (unsigned int face_batch = 0;
         face_batch < this->data->n_inner_face_batches();
         ++face_batch)
      {
        const auto initialize_face = [&](auto &phi, bool interior) {
          phi.reinit(face_batch);

          const unsigned int active_entries =
            this->data->n_active_entries_per_face_batch(face_batch);

          std::array<unsigned, n_lanes> cell_indices = phi.get_cell_ids();

          std::vector<Point<dim>> local_closest;
          local_closest.resize(n_quadrature_points);

          // for every point find the closest one
          for (unsigned lane = 0; lane < active_entries; ++lane)
            {
              auto cell_iter =
                this->data->get_cell_iterator(cell_indices[lane] / n_lanes,
                                              cell_indices[lane] % n_lanes);

              auto [cell_it, face_no] =
                this->data->get_face_iterator(face_batch, lane, interior);

              Assert(cell_iter == cell_it,
                     ExcMessage("Cell iterator mismatch"));

              for (const unsigned int q : phi.quadrature_point_indices())
                {
                  Point<dim, VectorizedNumber> q_point_vectorized =
                    phi.quadrature_point(q);

                  // unpack point
                  Point<dim> q_point;
                  for (unsigned d = 0; d < dim; ++d)
                    q_point(d) = q_point_vectorized(d)[lane];

                  face_quad_points[q] = q_point;
                }

              auto [closest_real_points, closest_unit_reference_points] =
                shifts_generator.generate_shifts(cell_it->neighbor(face_no),
                                                 cell_it,
                                                 face_quad_points);

              cells_vector.push_back(cell_iter);
              closest_points_vector.push_back(closest_unit_reference_points);
              n_sbm_faces++;
            }


          faces_offsets[face_batch] = current_offset;
          current_offset +=
            this->data->n_active_entries_per_face_batch(face_batch);
        };

        const std::pair<unsigned int, unsigned int> cell_cat =
          this->data->get_face_category(face_batch);

        if (is_cell_inside(cell_cat.first) && is_cell_outside(cell_cat.second))
          initialize_face(phi_inner, true);
        else if (is_cell_outside(cell_cat.first) &&
                 is_cell_inside(cell_cat.second))
          initialize_face(phi_outer, false);
        else
          continue;
      }

    surface_mapping_info =
      std::make_shared<NonMatching::MappingInfo<dim, dim, Number>>(
        mapping, update_values | update_quadrature_points);
    surface_mapping_info->reinit_cells(cells_vector, closest_points_vector);

    shifts_initialized = true;

    mpi_communicator =
      this->data->get_vector_partitioner()->get_mpi_communicator();
  }



  template <typename OPERATOR, typename VECTOR>
  void
  MatrixFreeOperator<OPERATOR, VECTOR>::output_shifts()
  {
    std::vector<std::pair<Point<dim>, Point<dim>>>          shifts;
    FEFaceEvaluation<dim, fe_degree, n_q_points, 1, Number> phi_inner(
      *this->data, true);
    FEFaceEvaluation<dim, fe_degree, n_q_points, 1, Number> phi_outer(
      *this->data, false);

    FEPointEvaluation<n_components, dim, dim, Number> shifted_phi(
      *surface_mapping_info, this->data->get_dof_handler(0).get_fe());

    for (unsigned int face_batch = 0;
         face_batch < this->data->n_inner_face_batches();
         ++face_batch)
      {
        const auto do_face = [&](auto &phi) {
          const auto offset = faces_offsets[face_batch];
          Assert(offset != numbers::invalid_unsigned_int, ExcInternalError());
          phi.reinit(face_batch);

          const unsigned int active_entries =
            this->data->n_active_entries_per_face_batch(face_batch);
          for (unsigned lane = 0; lane < active_entries; ++lane)
            {
              shifted_phi.reinit(offset + lane);

              for (const unsigned int q : phi.quadrature_point_indices())
                {
                  Point<dim> base_point;
                  for (unsigned d = 0; d < dim; d++)
                    base_point(d) = phi.quadrature_point(q)(d)[lane];

                  shifts.push_back(
                    std::make_pair(base_point,
                                   shifted_phi.quadrature_point(q)));
                }
            }
        };

        const std::pair<unsigned int, unsigned int> cell_cat =
          this->data->get_face_category(face_batch);
        if (is_cell_inside(cell_cat.first) && is_cell_outside(cell_cat.second))
          do_face(phi_inner);
        else if (is_cell_outside(cell_cat.first) &&
                 is_cell_inside(cell_cat.second))
          do_face(phi_outer);
        else
          continue;
      }

    const unsigned int this_mpi_process =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_mpi_processes =
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    ClosestPoint::output_shifts("shifts-MF2",
                                shifts,
                                n_mpi_processes,
                                this_mpi_process);
  }



  template <typename OPERATOR, typename VECTOR>
  inline std::array<double, 5>
  MatrixFreeOperator<OPERATOR, VECTOR>::get_timing(bool print) const
  {
    MPI_Comm mpi_comm             = mpi_communicator;
    double   global_interior_time = 0.0, global_interior_face_time = 0.0,
           global_boundary_face_time = 0.0, global_mpi_time = 0.0,
           global_total_time = 0.0;
    MPI_Allreduce(
      &interior_time, &global_interior_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&interior_face_time,
                  &global_interior_face_time,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_comm);
    MPI_Allreduce(&boundary_face_time,
                  &global_boundary_face_time,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_comm);
    MPI_Allreduce(
      &mpi_time, &global_mpi_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(
      &total_time, &global_total_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && print)
      {
        std::cout << "Interior time: " << global_interior_time << std::endl;
        std::cout << "Interior face time: " << global_interior_face_time
                  << std::endl;
        std::cout << "Boundary face time: " << global_boundary_face_time
                  << std::endl;
        std::cout << "MPI time: " << global_mpi_time << std::endl;
        // std::cout << "Total time: " << global_total_time << std::endl;
      }

    return std::array<double, 5>{{global_interior_time,
                                  global_interior_face_time,
                                  global_boundary_face_time,
                                  global_mpi_time,
                                  global_total_time}};
  }


  template <typename OPERATOR, typename VECTOR>
  inline void
  MatrixFreeOperator<OPERATOR, VECTOR>::apply_add(VectorType &      dst,
                                                  const VectorType &src) const
  {
    const auto &fe = this->data->get_dof_handler(0).get_fe();
    if (fe.conforms(FiniteElementData<dim>::Conformity::H1))
      do_interior_faces = false;

    // if (true)
    this->data->loop(&MatrixFreeOperator::local_apply,
                     &MatrixFreeOperator::apply_face,
                     &MatrixFreeOperator::apply_boundary,
                     this,
                     dst,
                     src);
    // else
    //   {
    //     local_apply(*this->data,
    //                 dst,
    //                 src,
    //                 std::make_pair(0, this->data->n_cell_batches()));
    //     apply_face(*this->data,
    //                dst,
    //                src,
    //                std::make_pair(0, this->data->n_inner_face_batches()));
    //   }

    // return;
  }



  template <typename OPERATOR, typename VECTOR>
  inline void
  MatrixFreeOperator<OPERATOR, VECTOR>::vmult(VectorType &      dst,
                                              const VectorType &src) const
  {
    // MatrixFreeOperators::Base<OPERATOR::dimension, VECTOR>::vmult(dst, src);
    // return;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
      this->data->get_vector_partitioner();

    // Assert(partitioner->is_globally_compatible(
    //          *data_reference->get_vector_partitioner().get()),
    //        ExcMessage("Current and reference partitioners are
    //        incompatible"));

    // adjust_ghost_range_if_necessary(partitioner, dst);
    // adjust_ghost_range_if_necessary(partitioner, src);
    const auto &fe = this->data->get_dof_handler(0).get_fe();
    if (fe.conforms(FiniteElementData<dim>::Conformity::H1))
      do_interior_faces = false;

    dst = 0;
    {
      Timer timer(mpi_time);

      // src.update_ghost_values();
    }
    local_apply(*this->data,
                dst,
                src,
                std::make_pair(0, this->data->n_cell_batches()));
    apply_face(*this->data,
               dst,
               src,
               std::make_pair(0, this->data->n_inner_face_batches()));
    {
      Timer timer(mpi_time);
      // src.zero_out_ghost_values();
      dst.compress(VectorOperation::add);
    }
  }



  template <typename OPERATOR, typename VECTOR>
  template <class OtherVector>
  std::vector<unsigned int>
  MatrixFreeOperator<OPERATOR, VECTOR>::generate_category_vector(
    const DoFHandler<dim> &dof_handler,
    const OtherVector &    level_set,
    const double           lambda,
    const unsigned int     level)
  {
    const unsigned int n_dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
    Vector<double> local_level_set_values(n_dofs_per_cell);


    if (level == numbers::invalid_unsigned_int)
      AssertDimension(level_set.size(), dof_handler.n_dofs());
    else
      AssertDimension(level_set.size(), dof_handler.n_dofs(level));

    const unsigned int n_cells =
      level == numbers::invalid_unsigned_int ?
        dof_handler.get_triangulation().n_active_cells() :
        dof_handler.get_triangulation().n_cells(level);

    const QGauss<1>              quad1d(2);
    cutCellTools::Generator<dim> quad_generator(dof_handler.get_fe(), quad1d);

    auto get_index = [&](const auto &cell) {
      return level == numbers::invalid_unsigned_int ?
               cell->active_cell_index() :
               cell->index();
    };

    std::vector<unsigned int> cell_vectorization_category(
      n_cells, numbers::invalid_unsigned_int);
    const auto generate = [&](auto begin_cell, const auto &endc) -> void {
      auto cell = begin_cell;
      for (; cell != endc; ++cell)
        {
          types::global_cell_index i = get_index(cell);
          AssertThrow(cell != endc, ExcInternalError());


          if (level == numbers::invalid_unsigned_int && cell->is_artificial())
            continue;
          if (level != numbers::invalid_unsigned_int &&
              cell->is_artificial_on_level())
            continue;


          if (level == numbers::invalid_unsigned_int)
            cell->get_dof_indices(dof_indices);
          else
            cell->get_mg_dof_indices(dof_indices);


          level_set.extract_subvector_to(dof_indices.begin(),
                                         dof_indices.end(),
                                         local_level_set_values.begin());

          bool all_positive = std::all_of(local_level_set_values.begin(),
                                          local_level_set_values.end(),
                                          [](double v) { return v > 0; });
          bool all_negative = std::all_of(local_level_set_values.begin(),
                                          local_level_set_values.end(),
                                          [](double v) { return v < 0; });
          if (all_positive)
            cell_vectorization_category[i] = CellStatus::cut;
          else if (all_negative)
            cell_vectorization_category[i] = CellStatus::inside;
          else if (!all_positive && !all_negative)
            {
              quad_generator.reinit(local_level_set_values);
              const Quadrature<dim> inside_quadrature =
                quad_generator.get_inside_quadrature();

              double inside_area = 0;
              for (const auto &weight : inside_quadrature.get_weights())
                inside_area += weight;

              if (inside_area > lambda)
                cell_vectorization_category[i] = CellStatus::inside;
              else
                cell_vectorization_category[i] = CellStatus::cut;
            }
          else
            Assert(false, ExcMessage("Invalid level set value"));

          // fixme: skip cells that are not locally owned
          // if(cell->is_locally_owned())
        }

      for (cell = begin_cell; cell != endc; ++cell)
        {
          const auto cell_index = get_index(cell);
          if (cell_vectorization_category[cell_index] != CellStatus::inside)
            continue;
          for (const auto &face : GeometryInfo<dim>::face_indices())
            {
              const auto neighbor = cell->neighbor(face);
              if (get_index(neighbor) == numbers::invalid_unsigned_int)
                continue;
              // fixme: skip cells that are not locally owned. Requires on level
              // etc.

              if (cell_vectorization_category[get_index(neighbor)] ==
                  CellStatus::cut)
                {
                  cell_vectorization_category[cell_index] =
                    CellStatus::intefacial;
                  break;
                }
            }
        }
    };

    if (level == numbers::invalid_unsigned_int)
      generate(dof_handler.begin_active(), dof_handler.end());
    else
      generate(dof_handler.begin(level), dof_handler.end(level));
    return cell_vectorization_category;
  }

  template <typename OPERATOR, typename VECTOR>
  void
  MatrixFreeOperator<OPERATOR, VECTOR>::compute_diagonal()
  {
    this->inverse_diagonal_entries =
      std::make_shared<DiagonalMatrix<VectorType>>();

    VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    this->data->loop(&MatrixFreeOperator::local_compute_cell_diagonal,
                     &MatrixFreeOperator::local_compute_face_diagonal,
                     &MatrixFreeOperator::local_compute_boundary_diagonal,
                     this,
                     inverse_diagonal,
                     (unsigned)0);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.size(); ++i)
      if (inverse_diagonal[i] != 0)
        inverse_diagonal[i] = 1. / inverse_diagonal[i];
  }



  template <typename OPERATOR, typename VECTOR>
  void
  MatrixFreeOperator<OPERATOR, VECTOR>::local_compute_cell_diagonal(
    const MatrixFree<dim, Number> &data,
    VectorType &                   dst,
    const unsigned int & /*component*/,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points, 1, Number> phi_cell(data, 0);
    AlignedVector<VectorizedNumber> diagonal(phi_cell.dofs_per_cell);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        const unsigned int cell_cat = data.get_cell_category(cell);
        if (is_cell_inside(cell_cat))
          {
            phi_cell.reinit(cell);
            for (unsigned int i = 0; i < phi_cell.dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < phi_cell.dofs_per_cell; ++j)
                  phi_cell.submit_dof_value(VectorizedArray<Number>(), j);
                phi_cell.submit_dof_value(make_vectorized_array<Number>(1.), i);

                //=============================
                phi_cell.evaluate(EvaluationFlags::gradients);

                for (const unsigned int q : phi_cell.quadrature_point_indices())
                  phi_cell.submit_gradient(phi_cell.get_gradient(q), q);

                phi_cell.integrate(EvaluationFlags::gradients);
                //=============================

                diagonal[i] = phi_cell.get_dof_value(i);
              }
            for (unsigned int i = 0; i < phi_cell.dofs_per_cell; ++i)
              phi_cell.submit_dof_value(diagonal[i], i);
            phi_cell.distribute_local_to_global(dst);
          }
        else if (cell_cat == CellStatus::cut)
          {}
        else if (cell_cat == CellStatus::outside)
          {}
        else
          {
            Assert(false, ExcNotImplemented());
          }
      }
  }



  template <typename OPERATOR, typename VECTOR>
  void
  MatrixFreeOperator<OPERATOR, VECTOR>::local_compute_face_diagonal(
    const MatrixFree<dim, Number> &data,
    VectorType &                   dst,
    const unsigned int & /*component*/,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    if (!faces_in_diagonal)
      return;


    Assert(shifts_initialized,
           ExcMessage(
             "Shifts not initialized. Call intialize_shifts() first."));
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(data,
                                                                         true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_outer(data,
                                                                         false);

    // fixme: hacked FEPointEvaluation
    FE_DGQ<dim>                                       fe_fake(fe_degree);
    FEPointEvaluation<n_components, dim, dim, Number> shifted_phi(
      *surface_mapping_info, fe_fake);

    AlignedVector<VectorizedNumber> diagonal(phi_inner.dofs_per_cell);


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        const auto evaluate = [&](auto &phi, const auto n_sign) {
          phi.reinit(face);
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.submit_dof_value(VectorizedArray<Number>(), j);
              phi.submit_dof_value(make_vectorized_array<Number>(1.), i);

              boundary_evaluation_kernel(phi, shifted_phi, n_sign);

              diagonal[i] = phi.get_dof_value(i);
            }

          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(dst);
        };

        const std::pair<unsigned int, unsigned int> cell_cat =
          data.get_face_category(face);

        // skip faces that are not cut
        if (is_cell_inside(cell_cat.first) && is_cell_outside(cell_cat.second))
          {
            evaluate(phi_inner, 1.);
          }
        else if (is_cell_outside(cell_cat.first) &&
                 is_cell_inside(cell_cat.second))
          {
            evaluate(phi_outer, -1.);
          }
        else
          continue;
      }
  }



  template <typename OPERATOR, typename VECTOR>
  std::pair<
    Table<3, typename MatrixFreeOperator<OPERATOR, VECTOR>::VectorizedNumber>,
    std::vector<unsigned int>>
  MatrixFreeOperator<OPERATOR, VECTOR>::assemble_interface_matrices(
    const SparseMatrix<double> &reference) const
  {
    std::vector<std::array<FullMatrix<Number>, n_lanes>> cell_matrices(
      this->data->n_cell_batches());
    std::vector<unsigned int> cell_offsets(this->data->n_cell_batches(),
                                           numbers::invalid_unsigned_int);
    unsigned int              current_offset = 0;

    const unsigned int n_dofs_per_cell =
      this->data->get_dof_handler(0).get_fe().dofs_per_cell;

    const auto distribute_column =
      [&](const unsigned int                column,
          std::array<unsigned int, n_lanes> cell_indices,
          ArrayView<VectorizedNumber>       cell_matrix_column,
          const unsigned int                n_entries) {
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
          {
            for (unsigned int lane = 0; lane < n_entries; ++lane)
              {
                const unsigned int cell_index = cell_indices[lane];

                if (cell_index == numbers::invalid_unsigned_int)
                  continue;

                const unsigned int cell_batch = cell_index / n_lanes;
                const unsigned int lane_index = cell_index % n_lanes;

                cell_matrices[cell_batch][lane_index](j, column) +=
                  cell_matrix_column[j][lane];
              }
          }
      };


    {
      FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> phi_cell(
        *this->data);

      for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
        {
          const auto cell_cat = this->data->get_cell_category(cell);
          if (cell_cat == CellStatus::intefacial)
            {
              const unsigned int n_active_lanes =
                this->data->n_active_entries_per_cell_batch(cell);
              for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                cell_matrices[cell][lane].reinit(n_dofs_per_cell,
                                                 n_dofs_per_cell);
              cell_offsets[cell] = current_offset;
              ++current_offset;


              phi_cell.reinit(cell);
              for (unsigned int i = 0; i < phi_cell.dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < phi_cell.dofs_per_cell; ++j)
                    phi_cell.submit_dof_value(VectorizedArray<Number>(), j);

                  phi_cell.submit_dof_value(VectorizedArray<Number>(1.), i);

                  phi_cell.evaluate(EvaluationFlags::gradients);
                  for (const unsigned int q :
                       phi_cell.quadrature_point_indices())
                    phi_cell.submit_gradient(phi_cell.get_gradient(q), q);
                  phi_cell.integrate(EvaluationFlags::gradients);

                  ArrayView<VectorizedNumber> cell_matrix_column(
                    phi_cell.begin_dof_values(), phi_cell.dofs_per_cell);

                  distribute_column(i,
                                    phi_cell.get_cell_ids(),
                                    cell_matrix_column,
                                    n_active_lanes);
                }
            }
        }
    }

    std::cout << " Assembling interface matrices" << std::endl;

    // evaluate face contributions
    Assert(shifts_initialized,
           ExcMessage(
             "Shifts not initialized. Call intialize_shifts() first."));

    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(
      *this->data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_outer(
      *this->data, false);

    // fixme: hacked FEPointEvaluation
    FE_DGQ<dim>                                       fe_fake(fe_degree);
    FEPointEvaluation<n_components, dim, dim, Number> shifted_phi(
      *surface_mapping_info, fe_fake);

    for (unsigned int face = 0; face < this->data->n_inner_face_batches();
         ++face)
      {
        const std::pair<unsigned int, unsigned int> cell_cat =
          this->data->get_face_category(face);

        const auto evaluate_boundary = [&](auto &phi, const auto n_sign) {
          phi.reinit(face);
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.submit_dof_value(VectorizedArray<Number>(), j);
              phi.submit_dof_value(make_vectorized_array<Number>(1.), i);

              boundary_evaluation_kernel(phi, shifted_phi, n_sign);

              ArrayView<VectorizedNumber> cell_matrix_column(
                phi.begin_dof_values(), phi.dofs_per_cell);

              distribute_column(i,
                                phi.get_cell_ids(),
                                cell_matrix_column,
                                n_lanes);
            }
        };


        const auto evaluate_face = [&](const unsigned int i, bool do_inside) {
          phi_inner.reinit(face);
          phi_outer.reinit(face);
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            {
              phi_inner.submit_dof_value(VectorizedNumber(), j);
              phi_outer.submit_dof_value(VectorizedNumber(), j);
            }
          if (do_inside)
            phi_inner.submit_dof_value(VectorizedNumber(1.), i);
          else
            phi_outer.submit_dof_value(VectorizedNumber(1.), i);

          // ========================
          internal_face_evaluation_kernel(phi_inner, phi_outer);
          // ========================

          ArrayView<VectorizedNumber> cell_matrix_column(
            do_inside ? phi_inner.begin_dof_values() :
                        phi_outer.begin_dof_values(),
            n_dofs_per_cell);

          distribute_column(i,
                            (do_inside ? phi_inner : phi_outer).get_cell_ids(),
                            cell_matrix_column,
                            n_lanes);
        };


        // skip face of which neither are interfacial
        if (cell_cat.first != CellStatus::intefacial &&
            cell_cat.second != CellStatus::intefacial)
          continue;

        if (is_cell_inside(cell_cat.first) && is_cell_outside(cell_cat.second))
          evaluate_boundary(phi_inner, 1.);
        else if (is_cell_outside(cell_cat.first) &&
                 is_cell_inside(cell_cat.second))
          evaluate_boundary(phi_outer, -1.);

        if (cell_cat.first == CellStatus::intefacial &&
            is_cell_inside(cell_cat.second))
          {
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              evaluate_face(i, true);
          }

        if (is_cell_inside(cell_cat.first) &&
            cell_cat.second == CellStatus::intefacial)
          {
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              evaluate_face(i, false);
          }
      }

    if (!reference.empty())
      for (unsigned int cell = 0; cell < cell_matrices.size(); ++cell)
        if (cell_matrices[cell][0].m() > 0 && cell_matrices[cell][0].n() > 0)
          for (unsigned int lane = 0; lane < n_lanes; ++lane)
            {
              if (cell_matrices[cell][lane].m() == 0 ||
                  cell_matrices[cell][lane].n() == 0)
                continue;

              const auto offset = cell_offsets[cell];
              Assert(offset != numbers::invalid_unsigned_int,
                     ExcInternalError());

              typename DoFHandler<dim>::active_cell_iterator cell_iter =
                this->data->get_cell_iterator(cell, lane);

              std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
              cell_iter->get_dof_indices(dof_indices);

              FullMatrix<double> reference_submatrix(n_dofs_per_cell,
                                                     n_dofs_per_cell);
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
                  reference_submatrix(i, j) =
                    reference.el(dof_indices[i], dof_indices[j]);

              const auto &       assembled_matrix = cell_matrices[cell][lane];
              FullMatrix<double> diff_matrix(n_dofs_per_cell, n_dofs_per_cell);
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
                  diff_matrix(i, j) =
                    assembled_matrix(i, j) - reference_submatrix(i, j);

              if (diff_matrix.frobenius_norm() > 1e-10)
                {
                  std::cout << "WARNING: Mismatch for cell "
                            << cell_iter->index() << " lane " << lane
                            << " diff norm: " << diff_matrix.frobenius_norm()
                            << std::endl;
                }
            }

    for (auto &cell_matrix_array : cell_matrices)
      for (auto &mat : cell_matrix_array)
        if (mat.m() > 0 && mat.n() > 0)
          mat.gauss_jordan();

    Table<3, VectorizedNumber> interface_inverses(current_offset,
                                                  n_dofs_per_cell,
                                                  n_dofs_per_cell);
    // Copy cell_matrices to interface_inverses
    for (unsigned int cell = 0; cell < cell_matrices.size(); ++cell)
      if (cell_matrices[cell][0].m() > 0 && cell_matrices[cell][0].n() > 0)
        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          {
            const auto &offset = cell_offsets[cell];
            Assert(offset != numbers::invalid_unsigned_int, ExcInternalError());

            if (cell_matrices[cell][lane].m() == 0 ||
                cell_matrices[cell][lane].n() == 0)
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
                  interface_inverses(offset, i, j)[lane] = 0.;
            else
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
                  interface_inverses(offset, i, j)[lane] =
                    cell_matrices[cell][lane](i, j);
          }

    return {interface_inverses, cell_offsets};
  }


  template <typename OPERATOR, typename VECTOR>
  std::size_t
  MatrixFreeOperator<OPERATOR, VECTOR>::memory_consumption_custom() const
  {
    MPI_Comm          mpi_comm = mpi_communicator;
    const std::size_t this_mem = this->data->memory_consumption() +
                                 surface_mapping_info->memory_consumption();
    std::size_t global_mem = 0;
    MPI_Allreduce(&this_mem, &global_mem, 1, MPI_UINT64_T, MPI_SUM, mpi_comm);
    return global_mem;
  }


  template <typename OPERATOR, typename VECTOR>
  class PreconditionBlockJacobiSBM
  {
  public:
    //  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
    typedef VECTOR                      VectorType;
    typedef OPERATOR                    LocalOperator;
    typedef typename VECTOR::value_type Number;
    typedef VectorizedArray<Number>     VectorizedNumber;

    const constexpr static int         dim       = OPERATOR::dimension;
    const constexpr static int         fe_degree = OPERATOR::fe_degree;
    const constexpr static std::size_t n_lanes   = VectorizedNumber::size();

    struct AdditionalData
    {
      AdditionalData(std::pair<Table<3, VectorizedNumber>,
                               std::vector<unsigned int>> &&interface_data = {},
                     const double                           relaxation = 1.0)
        : interface_data(std::move(interface_data))
        , relaxation(relaxation)
      {}

      std::pair<Table<3, VectorizedNumber>, std::vector<unsigned int>>
             interface_data;
      double relaxation;
    };

    PreconditionBlockJacobiSBM()
      : relaxation(1.0)
    {}

    PreconditionBlockJacobiSBM(
      const std::shared_ptr<const MatrixFree<dim, Number>> &mf_storage,
      AdditionalData &&additional_data = AdditionalData())
      : mf_storage(mf_storage)
      , interface_inverses(std::move(additional_data.interface_data.first))
      , cell_offsets(std::move(additional_data.interface_data.second))
      , relaxation(additional_data.relaxation)
    {
      initialize();
    }

    void
    initialize(const std::shared_ptr<const MatrixFree<dim, Number>> &mf_storage,
               const AdditionalData &additional_data = AdditionalData())
    {
      this->mf_storage         = mf_storage;
      this->interface_inverses = additional_data.interface_data.first;
      this->cell_offsets       = additional_data.interface_data.second;
      this->relaxation         = additional_data.relaxation;
      initialize();
    }

    template <typename MATRIX>
    void
    initialize(const MATRIX &        op,
               const AdditionalData &additional_data = AdditionalData())
    {
      initialize(op.get_matrix_free(), additional_data);
    }

    void
    clear()
    {
      mf_storage.reset();
      interface_inverses.reinit(0, 0, 0);
      cell_offsets.clear();
    }


    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0;
      step(dst, src);
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      vmult(dst, src);
    }

    void
    step(VectorType &dst, const VectorType &src) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_cell(
        *mf_storage);
      ArrayView<VectorizedNumber> phi_cell_values(phi_cell.begin_dof_values(),
                                                  phi_cell.dofs_per_cell);
      AlignedVector<VectorizedNumber> local_dst(phi_cell.dofs_per_cell);

      for (unsigned int cell = 0; cell < mf_storage->n_cell_batches(); ++cell)
        {
          const unsigned int cell_cat = mf_storage->get_cell_category(cell);
          if (cell_cat == CellStatus::intefacial)
            {
              const unsigned int offset = cell_offsets[cell];
              Assert(offset != numbers::invalid_unsigned_int,
                     ExcInternalError());
              phi_cell.reinit(cell);

              // read dof values
              phi_cell.read_dof_values(src);

              // apply cell inverse
              for (unsigned int i = 0; i < phi_cell.dofs_per_cell; ++i)
                {
                  VectorizedNumber sum = VectorizedNumber(0.);
                  for (unsigned int j = 0; j < phi_cell.dofs_per_cell; ++j)
                    sum +=
                      interface_inverses(offset, i, j) * phi_cell_values[j];
                  local_dst[i] = sum * relaxation;
                }

              // copy local_dst to phi_cell_values
              for (unsigned int i = 0; i < phi_cell.dofs_per_cell; ++i)
                phi_cell_values[i] = local_dst[i];

              // distribute to global vector
              phi_cell.distribute_local_to_global(dst);
              continue;
            }
          if (cell_cat == CellStatus::inside)
            {
              phi_cell.reinit(cell);

              // read dof values
              phi_cell.read_dof_values(src);

              cell_inverse.apply_inverse(local_dst, phi_cell_values);

              // copy local_dst to phi_cell_values
              for (unsigned int i = 0; i < phi_cell.dofs_per_cell; ++i)
                phi_cell_values[i] = local_dst[i] * relaxation;

              // distribute to global vector
              phi_cell.distribute_local_to_global(dst);
              continue;
            }
          if (cell_cat == CellStatus::cut || cell_cat == CellStatus::outside)
            {
              continue;
            }
        }
    }

    void
    Tstep(VectorType &dst, const VectorType &src) const
    {
      step(dst, src);
    }


  private:
    std::shared_ptr<const MatrixFree<dim, Number>> mf_storage;
    Table<3, VectorizedNumber> interface_inverses; // [cell][i][j][lane]
    std::vector<unsigned int>  cell_offsets;
    double                     relaxation;

    TensorProductMatrixSymmetricSum<dim, VectorizedNumber, fe_degree + 1>
      cell_inverse;

    void
    initialize()
    {
      auto   cell             = mf_storage->get_cell_iterator(0, 0);
      double cell_side_length = cell->minimum_vertex_distance();

      Assert(mf_storage != nullptr, ExcMessage("MatrixFree storage is null"));
      Assert(!interface_inverses.empty(),
             ExcMessage("Interface inverses are empty"));
      Assert(!cell_offsets.empty(), ExcMessage("Cell offsets are empty"));

      std::string name = mf_storage->get_dof_handler().get_fe().get_name();
      name.replace(name.find('<') + 1, 1, "1");
      std::unique_ptr<FiniteElement<1>> fe_1d =
        FETools::get_fe_by_name<1>(name);

      const unsigned int N = fe_degree + 1;
      FullMatrix<double> laplace(N, N);
      FullMatrix<double> mass(N, N);

      const QGauss<1> quadrature(N);
      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            double sum_mass = 0, sum_laplace = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_mass += (fe_1d->shape_value(i, quadrature.point(q)) *
                             fe_1d->shape_value(j, quadrature.point(q))) *
                            quadrature.weight(q);
                sum_laplace += (fe_1d->shape_grad(i, quadrature.point(q))[0] *
                                fe_1d->shape_grad(j, quadrature.point(q))[0]) *
                               quadrature.weight(q);
              }
            mass(i, j) = sum_mass;

            sum_laplace += (1. * fe_1d->shape_value(i, Point<1>()) *
                              fe_1d->shape_value(j, Point<1>()) *
                              get_penalty_factor(fe_degree) +
                            0.5 * fe_1d->shape_grad(i, Point<1>())[0] *
                              fe_1d->shape_value(j, Point<1>()) +
                            0.5 * fe_1d->shape_grad(j, Point<1>())[0] *
                              fe_1d->shape_value(i, Point<1>()));

            sum_laplace += (1. * fe_1d->shape_value(i, Point<1>(1.0)) *
                              fe_1d->shape_value(j, Point<1>(1.0)) *
                              get_penalty_factor(fe_degree) -
                            0.5 * fe_1d->shape_grad(i, Point<1>(1.0))[0] *
                              fe_1d->shape_value(j, Point<1>(1.0)) -
                            0.5 * fe_1d->shape_grad(j, Point<1>(1.0))[0] *
                              fe_1d->shape_value(i, Point<1>(1.0)));

            laplace(i, j) = sum_laplace;
          }

      const double h = cell_side_length;
      laplace *= 1.0 / h;
      mass *= h;

      Table<2, VectorizedNumber> vectorized_laplace(N, N);
      Table<2, VectorizedNumber> vectorized_mass(N, N);
      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            for (unsigned int lane = 0; lane < n_lanes; ++lane)
              {
                vectorized_laplace(i, j)[lane] = laplace(i, j);
                vectorized_mass(i, j)[lane]    = mass(i, j);
              }
          }
      cell_inverse.reinit(vectorized_mass, vectorized_laplace);
    }


    void
    adjust_ghost_range_if_necessary(const VectorType &vec) const
    {
      if (vec.get_partitioner().get() ==
          mf_storage->get_dof_info(0).vector_partitioner.get())
        return;

      VectorType copy_vec(vec);
      const_cast<VectorType &>(vec).reinit(
        mf_storage->get_dof_info(0).vector_partitioner);
      const_cast<VectorType &>(vec).copy_locally_owned_data_from(copy_vec);
    }
  };



} // namespace SBM
#endif /* INCLUDE_MATRIX_FREE_OPERATOR_H_ */
