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

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>
#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <tuple>

#include "cut_cell_generator.h"
#include "ghost_penalty_operator.h"



using namespace dealii;

template <class FEPointEvalType>
struct CutCellIntegrators
{
  template <class MappingInfoType, class Element>
  CutCellIntegrators(const MappingInfoType &          inside_mapping,
                     const MappingInfoType &          outside_mapping,
                     const MappingInfoType &          surface_mapping,
                     const Element &                  element,
                     const std::vector<unsigned int> &cut_cell_offsets)
    : n_dofs_per_cell(element.dofs_per_cell)
    , phi_inside(inside_mapping, element)
    , phi_outside(outside_mapping, element)
    , phi_surface(surface_mapping, element)
    , src_dst_lane(n_dofs_per_cell)
    , cut_cell_offsets(cut_cell_offsets)
    , cell_batch(numbers::invalid_unsigned_int)
    , lane_batch(numbers::invalid_unsigned_int)
    , lane_rw(numbers::invalid_unsigned_int)
    , degree(element.degree)
  {}

  using Number = typename FEPointEvalType::number_type;
  const constexpr static std::size_t n_lanes = VectorizedArray<Number>::size();

  const static unsigned int dim = FEPointEvalType::dimension;

  /*
   * This function is used to reinitialize the cut cell integrators. Because
   * we FEEvaluationcould reinitialized with std::array we allow providing the
   * lane index separately.
   *
   */
  template <int n_q_points_1d, int fe_degree, int n_components, typename Number>
  inline bool
  reinit(const FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>
           &                 phi,
         const unsigned int &lane_in)
  {
    const unsigned cell_index  = phi.get_cell_ids()[lane_in];
    cell_batch                 = cell_index / n_lanes;
    const unsigned int &offset = cut_cell_offsets.at(cell_batch);
    if (offset == numbers::invalid_unsigned_int)
      {
        lane_rw    = numbers::invalid_unsigned_int;
        lane_batch = numbers::invalid_unsigned_int;
        return false;
      }


    lane_batch = cell_index % n_lanes;
    lane_rw    = lane_in;
    phi_inside.reinit(offset + lane_batch);
    phi_outside.reinit(offset + lane_batch);
    phi_surface.reinit(offset + lane_batch);
    return true;
  }


  /*
   * This function is used to reinitialize the cut cell integrators and read the
   * values of the lane_in-th lane of the phi object.
   *
   * An exception is thrown if the reinitialization was not successful.
   */
  template <int n_q_points_1d, int fe_degree, int n_components, typename Number>
  inline void
  reinit_read_stride(
    const FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>
      &                 phi,
    const unsigned int &lane_in)
  {
    AssertDimension(fe_degree, degree);

    const bool succesful_reinit = reinit(phi, lane_in);
    Assert(
      succesful_reinit,
      ExcMessage(
        "CutCellIntegrators was not initialized successfully. Most likely it "
        " was initialized with cell for which data is not available"));

    read_stride(phi);
  }

  /*
   *
   *
   */
  inline void
  read_stride(const VectorizedArray<Number> *const begin_src)
  {
    Assert(lane_rw != numbers::invalid_unsigned_int,
           ExcMessage(
             "CutCellIntegrators was not initialized at all or "
             "it was initialized with cell for which data is not available"));

    StridedArrayView<const Number, n_lanes> strided_src_dst_view(
      &(begin_src[0][lane_rw]), n_dofs_per_cell);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      src_dst_lane[i] = strided_src_dst_view[i];
  }


  template <int n_q_points_1d, int fe_degree, int n_components, typename Number>
  inline void
  read_stride(
    const FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>
      &phi)
  {
    AssertDimension(fe_degree, degree);
    read_stride(phi.begin_dof_values());
  }

  inline void
  write_stride(VectorizedArray<Number> *begin_dst) const
  {
    Assert(lane_rw != numbers::invalid_unsigned_int,
           ExcMessage(
             "CutCellIntegrators was not initialized at all or "
             "it was initialized with cell for which data is not available"));
    StridedArrayView<Number, n_lanes> strided_src_dst_view(
      &begin_dst[0][lane_rw], n_dofs_per_cell);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      strided_src_dst_view[i] = src_dst_lane[i];
  }


  template <int n_q_points_1d, int fe_degree, int n_components, typename Number>
  inline void
  write_stride(FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>
                 &phi) const
  {
    AssertDimension(fe_degree, degree);

    write_stride(phi.begin_dof_values());
  }


  unsigned int                     n_dofs_per_cell;
  FEPointEvalType                  phi_inside;
  FEPointEvalType                  phi_outside;
  FEPointEvalType                  phi_surface;
  AlignedVector<Number>            src_dst_lane;
  const std::vector<unsigned int> &cut_cell_offsets;

  unsigned int cell_batch;
  unsigned int lane_batch;
  unsigned int lane_rw;

  const unsigned int degree;
};


template <typename Number>
void
adjust_ghost_range_if_necessary(
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
  const LinearAlgebra::distributed::Vector<Number> &        vec)
{
  if (vec.get_partitioner().get() != partitioner.get())
    {
      LinearAlgebra::distributed::Vector<Number> copy(vec);
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec).reinit(
        partitioner);
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec)
        .copy_locally_owned_data_from(copy);
    }
}



template <typename OPERATOR, typename VECTOR>
class MatrixFreeOperator : public EnableObserverPointer
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

  enum CellStatus
  {
    inside  = 0,
    cut     = 1,
    outside = 2
  };

  MatrixFreeOperator()
    // : MatrixFreeOperators::Base<dim, VectorType>()
    : skip_ghost_penalty(false)
    , ghost_penalaty_ready(false)
    , mpi_communicator(MPI_COMM_WORLD)
    , interior_time(0)
    , cut_time(0)
    , ghost_penalty_time(0)
    , mpi_time(0)
    , total_time(0)
  {
    selected_rows.resize(1);
    selected_rows[0] = 0;
    static_assert(
      std::is_same<typename LocalOperator::value_type, Number>::value);
    local_operator = std::make_shared<LocalOperator>();
  }


  template <typename MATRIX>
  void
  assemble_matrix(MATRIX &                         matrix,
                  const FiniteElement<dim> &       fe,
                  const AffineConstraints<double> &constrains) const;

  virtual void
  apply_add(VectorType &dst, const VectorType &src) const;

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

  auto
  get_cut_integrators() const
  {
    const unsigned int dof_handler_index = 0;
    const auto &dof_handler    = this->data->get_dof_handler(dof_handler_index);
    const auto &finite_element = dof_handler.get_fe(0);
    return CutCellIntegrators<
      FEPointEvaluation<n_components, dim, dim, Number>>(*inside_mapping_info,
                                                         *outside_mapping_info,
                                                         *surface_mapping_info,
                                                         *fe_no_renumbering,
                                                         cut_cell_offsets);
  }

  void
  vmult_add(VectorType &dst, const VectorType &src) const;

  void
  Tvmult_add(VectorType &dst, const VectorType &src) const;

  void
  vmult(VectorType &dst, const VectorType &src) const;

  void
  Tvmult(VectorType &dst, const VectorType &src) const;

  unsigned int
  n() const;

  unsigned int
  m() const;

  std::size_t
  memory_consumption_geometry() const
  {
    MPI_Comm          mpi_comm = mpi_communicator;
    const std::size_t this_mem = data->memory_consumption() +
                                 inside_mapping_info->memory_consumption() +
                                 surface_mapping_info->memory_consumption() +
                                 outside_mapping_info->memory_consumption();
    std::size_t global_mem = 0;
    MPI_Allreduce(&this_mem, &global_mem, 1, MPI_UINT64_T, MPI_SUM, mpi_comm);
    return global_mem;
  }

  void
  initialize_dof_vector(VectorType &vec) const;

  void
  initialize(
    std::shared_ptr<const MatrixFree<dim, Number, VectorizedNumber>> data);

  auto &
  get_matrix_free() const;

  std::array<double, 5>
  get_timing(bool print) const;

  template <class OtherVector>
  void
  initilize_cut_cells_quadrature(
    const hp::QCollection<1> &          quadratures1D,
    const DoFHandler<dim> &             dof_handler,
    const OtherVector &                 level_set,
    const Mapping<dim> &                mapping,
    const UpdateFlags                   inside_update_flags,
    const UpdateFlags                   outside_update_flags,
    const UpdateFlags                   surface_update_flags,
    const std::vector<Quadrature<dim>> &quadratures);

  template <class OtherVector>
  void
  initilize_cut_cells_quadrature(const Quadrature<1> &  quad_1D,
                                 const DoFHandler<dim> &dof_handler,
                                 const OtherVector &    level_set,
                                 const Mapping<dim> &   mapping,
                                 const UpdateFlags      inside_update_flags,
                                 const UpdateFlags      outside_update_flags,
                                 const UpdateFlags      surface_update_flags);

  void
  initialize_ghost_penalty_faces(
    const std::function<bool(const typename Triangulation<dim>::cell_iterator &,
                             const unsigned int)> &face_has_ghost_penalty =
      nullptr);

  template <typename OtherVector>
  static std::vector<unsigned int>
  generate_category_vector(
    const DoFHandler<dim> &dof_handler,
    const OtherVector &    level_set,
    const unsigned int     level = numbers::invalid_unsigned_int);



private:
  void
  local_apply(const MatrixFree<dim, Number> &              data,
              VectorType &                                 dst,
              const VectorType &                           src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;



  void
  local_compute_cell_diagonal(
    const MatrixFree<dim, Number> &              data,
    VectorType &                                 dst,
    const unsigned int &                         component,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  local_compute_boundary_diagonal(
    const MatrixFree<dim, Number> &              data,
    VectorType &                                 dst,
    const unsigned int &                         component,
    const std::pair<unsigned int, unsigned int> &face_range) const;

  void
  compute_face_diagonal(const MatrixFree<dim, Number> &data,
                        VectorType &                   dst,
                        const unsigned int &           component) const;


  /* not implemented*/
  void
  do_cut_cell_operation(
    FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> phi_cell,
    FEPointEvaluation<n_components, dim, dim, Number> &phi_cut_inside,
    FEPointEvaluation<n_components, dim, dim, Number> &phi_cut_outside,
    FEPointEvaluation<n_components, dim, dim, Number> &phi_cut_surface,
    const unsigned int &                               cell_batch) const;


  using VectorizedFacePair = std::pair<std::array<unsigned int, n_lanes>,
                                       std::array<unsigned int, n_lanes>>;

  // [direction][PairIndex]
  std::array<std::vector<VectorizedFacePair>, dim> ghost_penalty_faces;
  //[direction][cell]
  std::array<std::array<std::vector<unsigned int>, 2>, dim> cell2pair_numbering;
  //[direction]
  std::array<std::vector<unsigned int>, dim> duplicated_dofs;

  unsigned int n_dofs_per_face;

  std::unique_ptr<GhostPenalty::TensorProductApplier<dim, VectorizedNumber>>
                                 ghost_penalty_applier;
  std::shared_ptr<LocalOperator> local_operator;
  std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>
    inside_mapping_info;
  std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>
    surface_mapping_info;
  std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>>
    outside_mapping_info;

  std::vector<unsigned int> cut_cell_offsets;

  double cell_side_length;
  bool   skip_ghost_penalty;
  bool   ghost_penalaty_ready;

  std::shared_ptr<const MatrixFree<dim, Number, VectorizedNumber>> data;
  std::vector<unsigned int> selected_rows;

  std::unique_ptr<FiniteElement<dim>> fe_no_renumbering;

  MPI_Comm mpi_communicator;

  mutable double interior_time;
  mutable double cut_time;
  mutable double ghost_penalty_time;
  mutable double mpi_time;
  mutable double total_time;
};


template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::vmult_add(VectorType &      dst,
                                                const VectorType &src) const
{
  this->interior_time      = 0.0;
  this->cut_time           = 0.0;
  this->ghost_penalty_time = 0.0;
  this->mpi_time           = 0.0;
  this->total_time         = 0.0;

  auto start_time = std::chrono::high_resolution_clock::now();

  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
    this->data->get_vector_partitioner();

  // Assert(partitioner->is_globally_compatible(
  //          *data_reference->get_vector_partitioner().get()),
  //        ExcMessage("Current and reference partitioners are incompatible"));

  adjust_ghost_range_if_necessary(partitioner, dst);
  adjust_ghost_range_if_necessary(partitioner, src);

  {
    auto start_time_mpi = std::chrono::high_resolution_clock::now();
    src.update_ghost_values();
    mpi_time += std::chrono::duration<double>(
                  std::chrono::high_resolution_clock::now() - start_time_mpi)
                  .count();
  }

  // 2. loop over all locally owned cell blocks
  apply_add(dst, src);

  // 3. communicate results with MPI
  {
    auto start_time_mpi = std::chrono::high_resolution_clock::now();
    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
    mpi_time += std::chrono::duration<double>(
                  std::chrono::high_resolution_clock::now() - start_time_mpi)
                  .count();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  total_time += std::chrono::duration<double>(end_time - start_time).count();
  // // 4. constraints
  // for (const auto dof : this->data->get_constrained_dofs())
  //   dst.local_element(dof) += src.local_element(dof);
}

template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::Tvmult_add(VectorType &      dst,
                                                 const VectorType &src) const
{
  vmult_add(dst, src);
}

template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::vmult(VectorType &      dst,
                                            const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
}


template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::Tvmult(VectorType &      dst,
                                             const VectorType &src) const
{
  dst = 0;
  Tvmult_add(dst, src);
}


template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::initialize(
  std::shared_ptr<const MatrixFree<dim, Number, VectorizedNumber>> data)
{
  this->data = data;

  const auto &dof_handler = data->get_dof_handler(0);
  const auto &fe          = dof_handler.get_fe();
  Assert(dynamic_cast<const FE_Q<dim> *>(&fe) != nullptr,
         ExcMessage("MatrixFreeOperator only works with FE_Q elements"));
  this->fe_no_renumbering =
    std::make_unique<FE_DGQ<dim>>(data->get_dof_handler(0).get_fe().degree);

  mpi_communicator = data->get_vector_partitioner()->get_mpi_communicator();

  this->interior_time      = 0.0;
  this->cut_time           = 0.0;
  this->ghost_penalty_time = 0.0;
  this->mpi_time           = 0.0;
  this->total_time         = 0.0;
}

template <typename OPERATOR, typename VECTOR>
inline unsigned int
MatrixFreeOperator<OPERATOR, VECTOR>::n() const
{
  return this->data->get_dof_handler(0).n_dofs();
}

template <typename OPERATOR, typename VECTOR>
inline unsigned int
MatrixFreeOperator<OPERATOR, VECTOR>::m() const
{
  return this->data->get_dof_handler(0).n_dofs();
}

template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::initialize_dof_vector(
  VectorType &vec) const
{
  this->data->initialize_dof_vector(vec, this->selected_rows[0]);
}

template <typename OPERATOR, typename VECTOR>
inline auto &
MatrixFreeOperator<OPERATOR, VECTOR>::get_matrix_free() const
{
  return this->data;
}

template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::local_apply(
  const MatrixFree<dim, Number> &              data,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> phi_cell(data,
                                                                          0);

  auto cut_cell_integrators = get_cut_integrators();

  // const std::vector<unsigned int> dof_numbering =
  //   phi_cell.get_internal_dof_numbering();

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      const unsigned int cell_cat = data.get_cell_category(cell);
      if (cell_cat == CellStatus::inside)
        {
          auto start_time = std::chrono::high_resolution_clock::now();

          phi_cell.reinit(cell);
          phi_cell.read_dof_values(src);
          local_operator->integrate_cell(phi_cell, cell);
          phi_cell.distribute_local_to_global(dst);

          auto end_time = std::chrono::high_resolution_clock::now();
          interior_time +=
            std::chrono::duration<double>(end_time - start_time).count();
        }
      else if (cell_cat == CellStatus::cut)
        {
          auto start_time = std::chrono::high_resolution_clock::now();

          phi_cell.reinit(cell);
          phi_cell.read_dof_values(src);

          for (unsigned int lane = 0;
               lane < this->data->n_active_entries_per_cell_batch(cell);
               ++lane)
            {
              cut_cell_integrators.reinit_read_stride(phi_cell, lane);
              local_operator->integrate_cut_cell(cut_cell_integrators);
              cut_cell_integrators.write_stride(phi_cell);
            }
          phi_cell.distribute_local_to_global(dst);

          auto end_time = std::chrono::high_resolution_clock::now();
          cut_time +=
            std::chrono::duration<double>(end_time - start_time).count();
        }
      else if (cell_cat == CellStatus::outside)
        {}
      else
        {
          Assert(false, ExcNotImplemented());
        }
    }
}

template <typename OPERATOR, typename VECTOR>
inline std::array<double, 5>
MatrixFreeOperator<OPERATOR, VECTOR>::get_timing(bool print) const
{
  MPI_Comm mpi_comm             = mpi_communicator;
  double   global_interior_time = 0.0, global_cut_time = 0.0,
         global_ghost_penalty_time = 0.0, global_mpi_time = 0.0,
         global_total_time = 0.0;
  MPI_Allreduce(
    &interior_time, &global_interior_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
  MPI_Allreduce(&cut_time, &global_cut_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
  MPI_Allreduce(&ghost_penalty_time,
                &global_ghost_penalty_time,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                mpi_comm);
  MPI_Allreduce(&mpi_time, &global_mpi_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
  MPI_Allreduce(
    &total_time, &global_total_time, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0 && print)
    {
      std::cout << "Interior time: " << global_interior_time << std::endl;
      std::cout << "Cut time: " << global_cut_time << std::endl;
      std::cout << "Ghost penalty time: " << global_ghost_penalty_time
                << std::endl;
      std::cout << "MPI time: " << global_mpi_time << std::endl;
      std::cout << "Total time: " << global_total_time << std::endl;
    }

  return std::array<double, 5>{global_interior_time,
                               global_cut_time,
                               global_ghost_penalty_time,
                               global_mpi_time,
                               global_total_time};
}


template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::local_compute_cell_diagonal(
  const MatrixFree<dim, Number> &              data,
  VectorType &                                 dst,
  const unsigned int &                         component,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  (void)component;

  // FIXme: will not work for multicomponent problems
  AssertDimension(this->selected_rows.size(), 1);
  FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> phi(
    data, this->selected_rows[0]);


  auto        cut_cell_integrators = get_cut_integrators();
  const auto &n_dofs_per_cell      = cut_cell_integrators.n_dofs_per_cell;

  AlignedVector<VectorizedArray<Number>> diagonal(n_dofs_per_cell);
  diagonal.fill(0);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // Assert(range_cat == data.get_cell_category(cell),
      //        ExcDimensionMismatch(range_cat, data.get_cell_category(cell)));
      const unsigned int range_cat = data.get_cell_category(cell);

      phi.reinit(cell);
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = 0;
          phi.begin_dof_values()[i] = 1;


          if (range_cat == CellStatus::inside)
            local_operator->integrate_cell(phi, cell);
          else if (range_cat == CellStatus::cut)
            {
              for (unsigned int lane = 0;
                   lane < this->data->n_active_entries_per_cell_batch(cell);
                   ++lane)
                {
                  cut_cell_integrators.reinit_read_stride(phi, lane);
                  local_operator->integrate_cut_cell(cut_cell_integrators);
                  cut_cell_integrators.write_stride(phi);
                }
            }
          else if (range_cat == CellStatus::outside)
            {}
          else
            Assert(false, ExcNotImplemented());


          diagonal[i] = phi.get_dof_value(i);
        }
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        {
          for (unsigned int lane = 0; lane < n_lanes; ++lane)
            Assert(dealii::numbers::is_finite(diagonal[i][lane]),
                   ExcMessage(
                     "Diagonal is not finite, cell with  cat " +
                     std::to_string(range_cat) + "batch" +
                     std::to_string(cell) + " lane:" + std::to_string(lane) +
                     "cells in batch:  " +
                     std::to_string(
                       this->data->n_active_entries_per_cell_batch(cell))));
          phi.begin_dof_values()[i] = diagonal[i];
        }


      if (range_cat < 2)
        phi.distribute_local_to_global(dst);
    }
}

template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::compute_face_diagonal(
  const MatrixFree<dim, Number> &data,
  VectorType &                   dst,
  const unsigned int &           component) const
{
  (void)component;

  AssertDimension(this->selected_rows.size(), 1);
  FEEvaluation<dim, fe_degree, n_q_points, /* n_components */ 1, Number>
    phi_local(data, this->selected_rows[0]);
  FEEvaluation<dim, fe_degree, n_q_points, /* n_components */ 1, Number>
    phi_neighbor(data, this->selected_rows[0]);


  const auto assign = [](VectorizedNumber &left,
                         VectorizedNumber &right,
                         const bool &      to_face) -> void {
    if (to_face == 1)
      left = right;
    else
      right = left;
  };

  AlignedVector<VectorizedNumber> face_src(n_dofs_per_face);
  AlignedVector<VectorizedNumber> face_dst(n_dofs_per_face);
  for (unsigned int dir = 0; dir < dim; ++dir)
    {
      const auto cell2face = [&](const ArrayView<VectorizedNumber> &face_src,
                                 const bool &to_face) -> void {
        for (unsigned int c = 0; c < 2; ++c)
          for (unsigned int i = 0; i < phi_local.dofs_per_cell; ++i)
            assign(face_src[cell2pair_numbering[dir][c][i]],
                   (c == 0 ? phi_local.begin_dof_values()[i] :
                             phi_neighbor.begin_dof_values()[i]),
                   to_face);

        if (!to_face)
          {
            for (unsigned int dof : duplicated_dofs[dir])
              phi_neighbor.begin_dof_values()[dof] = 0;
          }
      };

      for (const auto &face_pair : ghost_penalty_faces[dir])
        {
          phi_local.reinit(face_pair.first);
          phi_neighbor.reinit(face_pair.second);

          for (unsigned int i = 0; i < n_dofs_per_face; ++i)
            {
              for (unsigned int j = 0; j < n_dofs_per_face; ++j)
                face_src[j] = 0;
              face_src[i] = 1;

              ghost_penalty_applier->vmult(face_dst, face_src);

              for (unsigned int j = 0; j < n_dofs_per_face; ++j)
                face_dst[j] = (i == j) ? face_dst[j] : 0;

              cell2face(face_dst, false);

              phi_local.distribute_local_to_global(dst);
              phi_neighbor.distribute_local_to_global(dst);
            }
        }
    }
}


template <typename OPERATOR, typename VECTOR>
inline void
MatrixFreeOperator<OPERATOR, VECTOR>::apply_add(VectorType &      dst,
                                                const VectorType &src) const
{
  // this->data->cell_loop(&MatrixFreeOperator::local_apply, this, dst, src);

  local_apply(*this->data,
              dst,
              src,
              std::make_pair<unsigned int, unsigned int>(
                0, this->data->n_cell_batches()));

  if (skip_ghost_penalty)
    return;

  auto start_time = std::chrono::high_resolution_clock::now();

  AlignedVector<VectorizedNumber> face_src(n_dofs_per_face);
  AlignedVector<VectorizedNumber> face_dst(n_dofs_per_face);

  FEEvaluation<dim, fe_degree, n_q_points, /* n_components */ 1, Number>
    phi_local(*this->data, this->selected_rows[0]);
  FEEvaluation<dim, fe_degree, n_q_points, /* n_components */ 1, Number>
    phi_neighbor(*this->data, this->selected_rows[0]);

  const unsigned int n_dofs_per_cell = phi_local.dofs_per_cell;

  std::array<ArrayView<VectorizedNumber>, 2> local_src;

  local_src[0] =
    ArrayView<VectorizedNumber>(phi_local.begin_dof_values(), n_dofs_per_cell);
  local_src[1] = ArrayView<VectorizedNumber>(phi_neighbor.begin_dof_values(),
                                             n_dofs_per_cell);



  const auto assign = [](VectorizedNumber &left,
                         VectorizedNumber &right,
                         const bool &      to_face) -> void {
    if (to_face == 1)
      left = right;
    else
      right = left;
  };


  for (unsigned int dir = 0; dir < dim; ++dir)
    {
      const auto cell2face =
        [&](const std::array<ArrayView<VectorizedNumber>, 2> &loc_src,
            const ArrayView<VectorizedNumber> &               face_src,
            const bool &                                      to_face) -> void {
        //
        for (unsigned int c = 0; c < 2; ++c)
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            assign(face_src[cell2pair_numbering[dir][c][i]],
                   loc_src[c][i],
                   to_face);

        // zero out dofs that are duplicated while writng back
        if (!to_face)
          {
            for (unsigned int dof : duplicated_dofs[dir])
              loc_src[1][dof] = 0;
          }
      };



      for (const auto &face_pair : ghost_penalty_faces[dir])
        {
          phi_local.reinit(face_pair.first);
          phi_neighbor.reinit(face_pair.second);

          phi_local.read_dof_values(src);
          phi_neighbor.read_dof_values(src);

          cell2face(local_src, face_src, true);
          // apply face ghost penaly
          // TODO
          Assert(ghost_penalaty_ready,
                 ExcMessage("Ghost penalty not initialized"));
          ghost_penalty_applier->vmult(face_dst, face_src);

          // write back to cell iterators
          cell2face(local_src, face_dst, false);

          phi_local.distribute_local_to_global(dst);
          phi_neighbor.distribute_local_to_global(dst);
        }
    }

  auto end_time = std::chrono::high_resolution_clock::now();
  ghost_penalty_time +=
    std::chrono::duration<double>(end_time - start_time).count();
}


template <typename OPERATOR, typename VECTOR>
template <class OtherVector>
void
MatrixFreeOperator<OPERATOR, VECTOR>::initilize_cut_cells_quadrature(
  const Quadrature<1> &  quad_1D,
  const DoFHandler<dim> &dof_handler,
  const OtherVector &    level_set,
  const Mapping<dim> &   mapping,
  const UpdateFlags      inside_update_flags,
  const UpdateFlags      outside_update_flags,
  const UpdateFlags      surface_update_flags)
{
  initilize_cut_cells_quadrature(hp::QCollection<1>(quad_1D),
                                 dof_handler,
                                 level_set,
                                 mapping,
                                 inside_update_flags,
                                 outside_update_flags,
                                 surface_update_flags,
                                 std::vector<Quadrature<dim>>());
}


template <typename OPERATOR, typename VECTOR>
template <class OtherVector>
void
MatrixFreeOperator<OPERATOR, VECTOR>::initilize_cut_cells_quadrature(
  const hp::QCollection<1> &          quadratures1D,
  const DoFHandler<dim> &             dof_handler,
  const OtherVector &                 level_set,
  const Mapping<dim> &                mapping,
  const UpdateFlags                   inside_update_flags,
  const UpdateFlags                   outside_update_flags,
  const UpdateFlags                   surface_update_flags,
  const std::vector<Quadrature<dim>> &quadratures)
{
  // NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
  //   quadratures1D, dof_handler, level_set);

  cut_cell_offsets.clear();
  cut_cell_offsets.resize(this->data->n_cell_batches(),
                          numbers::invalid_unsigned_int);

  const Quadrature<1>          quad1d = quadratures1D[0];
  cutCellTools::Generator<dim> quad_generator(dof_handler.get_fe(),
                                              quadratures1D[0]);
  const unsigned int n_dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
  Vector<double>                       local_level_set_values(n_dofs_per_cell);



  const unsigned int cut_cell_category = 1;

  AssertDimension(this->selected_rows.size(), 1);

  // Quadrature<dim> dummy_quad;
  std::vector<typename Triangulation<dim, dim>::cell_iterator> cells_vector;
  std::vector<Quadrature<dim>> inside_quadrature_vector;
  std::vector<Quadrature<dim>> outside_quadrature_vector;
  std::vector<NonMatching::ImmersedSurfaceQuadrature<dim>>
    surface_quadrature_vector;

  unsigned int current_offset = 0;
  for (unsigned int cell_b = 0; cell_b < this->data->n_cell_batches(); ++cell_b)
    {
      if (this->data->get_cell_category(cell_b) != cut_cell_category)
        continue;

      const unsigned int cell_cat = this->data->get_cell_category(cell_b);
      Assert(CellStatus::cut == cell_cat, ExcDimensionMismatch(cell_cat, 1));

      for (unsigned int lane = 0;
           lane < this->data->n_active_entries_per_cell_batch(cell_b);
           ++lane)
        {
          auto cell = this->data->get_cell_iterator(cell_b, lane);
          typename DoFHandler<dim>::cell_iterator cell_it(
            &cell->get_triangulation(),
            cell->level(),
            cell->index(),
            &dof_handler);
          if (this->data->get_mg_level() == numbers::invalid_unsigned_int)
            cell_it->get_dof_indices(dof_indices);
          else
            cell_it->get_mg_dof_indices(dof_indices);

          level_set.extract_subvector_to(dof_indices.begin(),
                                         dof_indices.end(),
                                         local_level_set_values.begin());

          quad_generator.reinit(local_level_set_values);


          this->cell_side_length = cell->minimum_vertex_distance();
          // quadrature_generator.generate(cell);
          // Even if a cell is formally intersected the number of created
          // quadrature points can be 0. Avoid creating an FEValues object
          // if that is the case.

          // fixme:
          const Quadrature<dim> &inside_quadrature =
            quad_generator.get_inside_quadrature();
          // fixme: should there really be a reference?
          const Quadrature<dim> &outside_quadrature =
            quad_generator.get_outside_quadrature();

          if (quadratures.size() != 0)
            Assert(quadratures[cell->user_index()] == inside_quadrature,
                   ExcMessage("Quadrature mismatch"));

          const NonMatching::ImmersedSurfaceQuadrature<dim>
            &surface_quadrature = quad_generator.get_surface_quadrature();
          if (inside_quadrature.size() == 0)
            {
              AssertThrow(true, ExcEmptyObject());
              // generate empty quadrature...
            }

          cells_vector.push_back(cell);
          inside_quadrature_vector.push_back(inside_quadrature);
          surface_quadrature_vector.push_back(surface_quadrature);
          outside_quadrature_vector.push_back(outside_quadrature);
        }
      cut_cell_offsets[cell_b] = current_offset;
      current_offset += this->data->n_active_entries_per_cell_batch(cell_b);
    }

  inside_mapping_info =
    std::make_shared<NonMatching::MappingInfo<dim, dim, Number>>(
      mapping, inside_update_flags);
  inside_mapping_info->reinit_cells(cells_vector, inside_quadrature_vector);

  outside_mapping_info =
    std::make_shared<NonMatching::MappingInfo<dim, dim, Number>>(
      mapping, outside_update_flags);
  outside_mapping_info->reinit_cells(cells_vector, outside_quadrature_vector);

  surface_mapping_info =
    std::make_shared<NonMatching::MappingInfo<dim, dim, Number>>(
      mapping, surface_update_flags);
  surface_mapping_info->reinit_surface(cells_vector, surface_quadrature_vector);


  // check if the quadrature points are the same
  if (quadratures.size() == 0)
    return;
  for (unsigned int inx = 0; inx < inside_quadrature_vector.size(); ++inx)
    {
      const auto &finite_element = dof_handler.get_fe(0);
      FEPointEvaluation<n_components, dim, dim, Number> phi_cut_cell(
        *inside_mapping_info, finite_element);
      phi_cut_cell.reinit(inx);
      const auto &quad0 = inside_quadrature_vector[inx];
      for (unsigned int q = 0; q < quad0.size(); ++q)
        Assert(phi_cut_cell.unit_point(q) == quad0.point(q),
               ExcMessage("Quadrature mismatch"));
    }
}

template <typename OPERATOR, typename VECTOR>
void
MatrixFreeOperator<OPERATOR, VECTOR>::initialize_ghost_penalty_faces(
  const std::function<bool(const typename Triangulation<dim>::cell_iterator &,
                           const unsigned int)> &face_has_ghost_penalty)
{
  const unsigned int dof_handler_index = 0;
  const auto &dof_handler    = this->data->get_dof_handler(dof_handler_index);
  const auto &finite_element = dof_handler.get_fe(0);



  std::map<unsigned int, unsigned int> face2direction;

  // Get tensor product numbering on faces
  for (unsigned int d = 0; d < dim; ++d)
    {
      GhostPenalty::Generator<dim> generator(finite_element.degree, d);
      for (unsigned int c = 0; c < 2; ++c)
        cell2pair_numbering[d][c] =
          generator.get_cell_tensor2face_tensor_numbering()[c];

      face2direction[generator.get_face()] = d;
      duplicated_dofs[d]                   = generator.get_duplicated_dofs();


      n_dofs_per_face = generator.n_interface_dofs();
      ghost_penalty_faces[d].clear();
    }

  // {cell, neighbor_index, direction}
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
    ghost_penalty_tuples;


  // fixme remove this one, we already have the cell_side_length
  double cell_side_length = 0;

  // Generate the map:
  for (unsigned int cell_b = 0; cell_b < this->data->n_cell_batches(); ++cell_b)
    for (unsigned int lane = 0;
         lane < this->data->n_active_entries_per_cell_batch(cell_b);
         ++lane)
      {
        auto cell        = this->data->get_cell_iterator(cell_b, lane);
        cell_side_length = cell->minimum_vertex_distance();

        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;

            const unsigned int neig_index =
              this->data->get_matrix_free_cell_index(cell->neighbor(f));
            AssertThrow(neig_index != numbers::invalid_unsigned_int,
                        ExcMessage("Invalid neighbor index"));

            const unsigned int neig_batch = neig_index / n_lanes;
            const unsigned int neig_lane  = neig_index % n_lanes;
            Assert(
              cell->neighbor(f)->index() ==
                this->data->get_cell_iterator(neig_batch, neig_lane)->index(),
              ExcMessage("Cell mismatch"));

            const unsigned int cell_cat = this->data->get_cell_category(cell_b);
            const unsigned int neig_cat =
              this->data->get_cell_category(neig_batch);

            bool has_ghost_penalty =
              (cell_cat != neig_cat || CellStatus::cut == neig_cat) &&
              cell_cat != CellStatus::outside &&
              neig_cat != CellStatus::outside;

            if (nullptr != face_has_ghost_penalty)
              Assert(has_ghost_penalty == face_has_ghost_penalty(cell, f),
                     ExcMessage(
                       "Ghost penalty mismatch: " + std::to_string(cell_cat) +
                       " vs " + std::to_string(neig_cat)));

            if (has_ghost_penalty)
              {
                const unsigned int index =
                  cell_b * VectorizedNumber::size() + lane;
                ghost_penalty_tuples.push_back(
                  std::make_tuple(index, neig_index, f));
              }
          }
      }

  // [direction][cell ( 0 or 1 ) ][index]
  std::array<VectorizedFacePair, dim> tmp_face_batch;
  std::array<unsigned int, dim>       face_batch_capacity;

  const auto reset = [&](const unsigned int &direction) -> void {
    for (unsigned int i = 0; i < n_lanes; ++i)
      {
        tmp_face_batch[direction].first[i]  = numbers::invalid_dof_index;
        tmp_face_batch[direction].second[i] = numbers::invalid_dof_index;
      }
    face_batch_capacity[direction] = 0;
  };

  const auto flush = [&](const unsigned int &direction) -> void {
    AssertIndexRange(direction, dim);
    ghost_penalty_faces[direction].push_back(tmp_face_batch[direction]);
    reset(direction);
  };

  const auto local_push_back = [&](const unsigned int &local_index,
                                   const unsigned int &neighbor_index,
                                   const unsigned int &direction) -> void {
    AssertIndexRange(direction, dim);
    const unsigned int batch_index = face_batch_capacity[direction];
    tmp_face_batch[direction].first[batch_index]  = local_index;
    tmp_face_batch[direction].second[batch_index] = neighbor_index;

    ++face_batch_capacity[direction];
    // if the batch is complete: flush.
    if (face_batch_capacity[direction] == n_lanes)
      flush(direction);
  };

  for (unsigned int d = 0; d < dim; ++d)
    reset(d);

  for (const auto &ghost_tuple : ghost_penalty_tuples)
    {
      const unsigned int local_index    = std::get<0>(ghost_tuple);
      const unsigned int neighbor_index = std::get<1>(ghost_tuple);
      const unsigned int face           = std::get<2>(ghost_tuple);
      // directions 1->0, 3 -> 1, 5->2
      for (const auto &fd : face2direction)
        if (fd.first == face)
          {
            const unsigned int direction = fd.second;
            local_push_back(local_index, neighbor_index, direction);
          }
    }

  for (unsigned int d = 0; d < dim; ++d)
    if (face_batch_capacity[d] != 0)
      flush(d);

  if (cell_side_length == 0)
    return;

  GhostPenalty::Generator1D generator(fe_degree);
  AssertIsFinite(cell_side_length);
  ghost_penalty_applier =
    std::make_unique<GhostPenalty::TensorProductApplier<dim, VectorizedNumber>>(
      generator.get_mass_matrix(cell_side_length),
      generator.get_penalty_matrix(cell_side_length));

  ghost_penalaty_ready = true;
}



template <typename OPERATOR, typename VECTOR>
template <class OtherVector>
std::vector<unsigned int>
MatrixFreeOperator<OPERATOR, VECTOR>::generate_category_vector(
  const DoFHandler<dim> &dof_handler,
  const OtherVector &    level_set,
  const unsigned int     level)
{
  const unsigned int n_dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
  Vector<double>                       local_level_set_values(n_dofs_per_cell);

  if (level == numbers::invalid_unsigned_int)
    AssertDimension(level_set.size(), dof_handler.n_dofs());
  else
    AssertDimension(level_set.size(), dof_handler.n_dofs(level));

  const unsigned int n_cells =
    level == numbers::invalid_unsigned_int ?
      dof_handler.get_triangulation().n_active_cells() :
      dof_handler.get_triangulation().n_cells(level);


  std::vector<unsigned int> cell_vectorization_category(
    n_cells, numbers::invalid_unsigned_int);
  const auto generate = [&](auto cell, const auto &endc) -> void {
    for (unsigned int i = 0; i < n_cells; ++i)
      {
        AssertThrow(cell != endc, ExcInternalError());

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
          cell_vectorization_category[i] = CellStatus::outside;
        else if (all_negative)
          cell_vectorization_category[i] = CellStatus::inside;
        else if (!all_positive && !all_negative)
          cell_vectorization_category[i] = CellStatus::cut;
        else
          Assert(false, ExcMessage("Invalid level set value"));

        ++cell;
        // fixme: skip cells that are not locally owned
        // if(cell->is_locally_owned())
      }
  };

  if (level == numbers::invalid_unsigned_int)
    generate(dof_handler.begin_active(), dof_handler.end());
  else
    generate(dof_handler.begin(level), dof_handler.end(level));
  return cell_vectorization_category;
}


#endif /* INCLUDE_MATRIX_FREE_OPERATOR_H_ */
