
// Compare FEPointEvaluation for scalar FE_Q to FEEvaluation.

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <iostream>

#include "ghost_penalty_operator.h"

using namespace dealii;


struct Timings
{
  double fe_point_evaluation;
  double fe_evaluation;
  double ghost_penalty;
  double sbm_face;
  double full_matrix;
};

struct MemoryUsage
{
  size_t fe_point_evaluation;
  size_t fe_evaluation;
  size_t ghost_penalty;
  size_t sbm_face;
  size_t full_matrix;
  size_t empty_mapping_info_values;
  size_t empty_mapping_info_grad;
};

template <int dim, unsigned int degree>
auto
test()
{
  using Number               = double;
  const unsigned int n_lanes = VectorizedArray<Number>::size();
  using VectorizedNumber     = VectorizedArray<Number>;
  Triangulation<dim> tria;

  GridGenerator::subdivided_hyper_cube(tria, dim == 2 ? 4 : 2, 0, 1);

  MappingQ<dim> mapping(std::max<unsigned int>(1, degree));

  QGauss<dim>        quadrature(degree + 1);
  QGauss<dim - 1>    face_quadrature(degree + 1);
  QGauss<1>          quad_1d(degree + 1);
  const unsigned int n_points      = quadrature.size();
  const unsigned int n_face_points = face_quadrature.size();

  FE_Q<dim>   fe(degree);
  FE_DGQ<dim> fe_dummy(degree);

  const unsigned int n_dofs_per_cell = fe.dofs_per_cell;

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  Vector<double> vector(dof_handler.n_dofs());


  NonMatching::MappingInfo<dim, dim> mapping_info(
    mapping, update_values | update_gradients | update_JxW_values);

  const auto cell = dof_handler.begin_active();
  mapping_info.reinit(cell, quadrature);

  std::vector<Point<dim>> local_closest_points;
  // Fill local_closest_points with random points, size n_face_points
  local_closest_points.resize(n_face_points);
  for (auto &pt : local_closest_points)
    for (unsigned int d = 0; d < dim; ++d)
      pt[d] = static_cast<double>(std::rand()) / RAND_MAX;
  NonMatching::MappingInfo<dim, dim> face_mapping_info(mapping, update_values);
  Quadrature<dim>                    face_points(local_closest_points);
  face_mapping_info.reinit(cell, face_points);

  NonMatching::MappingInfo<dim, dim> surface_mapping_info(
    mapping,
    update_values | update_gradients | update_JxW_values |
      update_normal_vectors);

  NonMatching::ImmersedSurfaceQuadrature<dim> surface_quad;
  for (unsigned int i = 0; i < n_face_points; ++i)
    {
      Point<dim>     pt;
      Tensor<1, dim> normal;
      for (unsigned int d = 0; d < dim; ++d)
        {
          pt[d]     = static_cast<double>(std::rand()) / RAND_MAX;
          normal[d] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;
        }
      // Normalize the normal vector
      normal /= normal.norm();
      double weight = static_cast<double>(std::rand()) / RAND_MAX;
      surface_quad.push_back(pt, weight, normal);
    }


  surface_mapping_info.reinit(cell, face_points);

  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_normal_vectors);


  std::shared_ptr<MatrixFree<dim, double>> mf_storage(
    new MatrixFree<dim, double>());

  AffineConstraints<double> constraints_dummy;
  mf_storage->reinit(
    mapping, dof_handler, constraints_dummy, quad_1d, additional_data);



  FEPointEvaluation<1, dim> point_evaluator(mapping_info, fe_dummy);
  FEPointEvaluation<1, dim> surface_evaluator(surface_mapping_info, fe_dummy);
  FEPointEvaluation<1, dim> shifted_phi(face_mapping_info, fe_dummy);

  FEEvaluation<dim, degree, degree + 1, 1, double> phi_cell(
    fe, quad_1d, update_values | update_gradients | update_JxW_values);

  FEFaceEvaluation<dim, degree, degree + 1, 1, Number> face_phi(*mf_storage,
                                                                true);

  Tensor<1, dim> exponents;
  exponents[0] = 1.;
  VectorTools::interpolate(mapping,
                           dof_handler,
                           Functions::Monomial<dim>(exponents),
                           vector);

  const unsigned int n_face_dofs =
    (2 * degree + 1) * (degree + 1) * (dim == 3 ? degree + 1 : 1);

  AlignedVector<double>           solution_values(fe.dofs_per_cell);
  AlignedVector<VectorizedNumber> dummy_src(fe.dofs_per_cell);
  AlignedVector<VectorizedNumber> dummy_dst(fe.dofs_per_cell);
  AlignedVector<VectorizedNumber> face_src(n_face_dofs);
  AlignedVector<VectorizedNumber> face_dst(n_face_dofs);
  AlignedVector<VectorizedNumber> shifted_values(n_face_points);
  for (auto &val : solution_values)
    val = std::rand() / (double)RAND_MAX;
  for (auto &val : face_src)
    val = std::rand() / (double)RAND_MAX;
  for (auto &val : face_dst)
    val = std::rand() / (double)RAND_MAX;

  point_evaluator.reinit();
  surface_evaluator.reinit();


  const unsigned int n_iter = 10000;
  Timings            timings;
  MemoryUsage        memory_usages;


  Quadrature<dim> dummy_quad;
  // ({{Point<dim>()}});
  NonMatching::MappingInfo<dim, dim> dummy_mapping_value(mapping,
                                                         update_values);
  NonMatching::MappingInfo<dim, dim> dummy_mapping_grad(
    mapping, update_values | update_gradients | update_JxW_values);

  dummy_mapping_value.reinit(cell, dummy_quad);
  dummy_mapping_grad.reinit(cell, dummy_quad);


  memory_usages.fe_point_evaluation = mapping_info.memory_consumption() +
                                      surface_mapping_info.memory_consumption();

  memory_usages.sbm_face = face_mapping_info.memory_consumption();

  memory_usages.empty_mapping_info_values =
    dummy_mapping_value.memory_consumption();
  memory_usages.empty_mapping_info_grad =
    dummy_mapping_grad.memory_consumption();


  {
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < n_iter; ++k)
      {
        point_evaluator.evaluate(solution_values, EvaluationFlags::gradients);
        surface_evaluator.evaluate(solution_values,
                                   EvaluationFlags::gradients |
                                     EvaluationFlags::values);
        for (unsigned int i = 0; i < n_points; ++i)
          point_evaluator.submit_gradient(point_evaluator.get_gradient(i), i);
        point_evaluator.integrate(solution_values,
                                  EvaluationFlags::gradients,
                                  false);


        for (const unsigned int q :
             surface_evaluator.quadrature_point_indices())
          {
            const Tensor<1, dim, Number> normal =
              surface_evaluator.normal_vector(q);
            const Number uh_minus_nGradU =
              -surface_evaluator.get_gradient(q) * normal +
              (0.42) * surface_evaluator.get_value(q);

            const Tensor<1, dim, Number> minus_nU =
              -surface_evaluator.get_value(q) * normal;

            surface_evaluator.submit_gradient(minus_nU, q);
            surface_evaluator.submit_value(uh_minus_nGradU, q);
          }


        surface_evaluator.integrate(solution_values,
                                    EvaluationFlags::gradients |
                                      EvaluationFlags::values,
                                    true);
      }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "FEPointEvaluation time: " << duration.count() / 1000. << "
    // ms"
    //           << std::endl;
    timings.fe_point_evaluation = duration.count() / (double)n_iter;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();

    const double normal_sign = 1.0;
    face_phi.reinit(0);
    for (unsigned int k = 0; k < n_iter; ++k)
      {
        for (unsigned lane = 0; lane < n_lanes; ++lane)
          {
            StridedArrayView<const Number, n_lanes> strided_src_dst_view(
              &(face_phi.begin_dof_values()[0][lane]), n_dofs_per_cell);
            shifted_phi.evaluate(strided_src_dst_view, EvaluationFlags::values);

            for (unsigned int q = 0; q < n_face_points; ++q)
              shifted_values[q][lane] = shifted_phi.get_value(q);
          }

        face_phi.evaluate(EvaluationFlags::gradients);

        for (const unsigned int q : face_phi.quadrature_point_indices())
          {
            face_phi.submit_value(0.5 * shifted_values[q] -
                                    normal_sign * face_phi.normal_vector(q) *
                                      face_phi.get_gradient(q),
                                  q);
            face_phi.submit_normal_derivative(-normal_sign * shifted_values[q],
                                              q);
          }
        face_phi.integrate(EvaluationFlags::values |
                           EvaluationFlags::gradients);
      }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "FEPointEvaluation time: " << duration.count() / 1000. << "
    // ms"
    //           << std::endl;
    timings.sbm_face = duration.count() / (double)n_iter;
  }


  {
    Table<2, VectorizedNumber> cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          cell_matrix[i][j][lane] = static_cast<double>(std::rand()) / RAND_MAX;

    memory_usages.full_matrix = cell_matrix.memory_consumption();

    auto start = std::chrono::high_resolution_clock::now();

    face_phi.reinit(0);
    for (unsigned int k = 0; k < n_iter; ++k)
      {
        // Matrix-vector multiplication: dummy_dst = cell_matrix * dummy_src
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            dummy_dst[i] = 0.0;
            for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
              dummy_dst[i] += cell_matrix[i][j] * dummy_src[j];
          }
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          dummy_src[i] = dummy_dst[i];
      }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    timings.full_matrix = duration.count() / (double)n_iter;
  }


  {
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < n_iter; ++k)
      {
        phi_cell.reinit(cell);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          phi_cell.begin_dof_values()[i] = solution_values[i];

        phi_cell.evaluate(EvaluationFlags::gradients);
        for (unsigned int i = 0; i < n_points; ++i)
          phi_cell.submit_gradient(point_evaluator.get_gradient(i), i);
        phi_cell.integrate(EvaluationFlags::gradients);
      }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "FEEvaluation time: " << duration.count() / 1000. << " ms"
    //           << std::endl;
    timings.fe_evaluation = duration.count() / (double)n_iter;
  }



  std::unique_ptr<GhostPenalty::TensorProductApplier<dim, VectorizedNumber>>
                            ghost_penalty_applier;
  GhostPenalty::Generator1D generator(degree);
  ghost_penalty_applier =
    std::make_unique<GhostPenalty::TensorProductApplier<dim, VectorizedNumber>>(
      generator.get_mass_matrix(1.), generator.get_penalty_matrix(1.));

  {
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < n_iter; ++k)
      {
        ghost_penalty_applier->vmult(face_src, face_dst);
        face_src = face_dst;
      }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    timings.ghost_penalty = duration.count() / (double)n_iter;
  }
  return std::pair(timings, memory_usages);
}



template <int dim, int degree, int max_degree>
void
run_tests(ConvergenceTable &timings_table, ConvergenceTable &memory_table)
{
  if constexpr (degree > max_degree)
    return;
  else
    {
      auto [timings, memory_usages] = test<dim, degree>();
      timings_table.add_value("degree", degree);
      timings_table.add_value("FEPointEvaluation", timings.fe_point_evaluation);
      timings_table.add_value("FEEvaluation", timings.fe_evaluation);
      timings_table.add_value("GhostPenalty", timings.ghost_penalty);
      timings_table.add_value("SBMFace", timings.sbm_face);
      timings_table.add_value("FullMatrix", timings.full_matrix);

      memory_table.add_value("degree", degree);
      memory_table.add_value("FEPointEvaluation",
                             memory_usages.fe_point_evaluation);
      memory_table.add_value("SBMFace", memory_usages.sbm_face);
      memory_table.add_value("FullMatrix", memory_usages.full_matrix);
      memory_table.add_value("EmptyMappingInfoValues",
                             memory_usages.empty_mapping_info_values);
      memory_table.add_value("EmptyMappingInfoGrad",
                             memory_usages.empty_mapping_info_grad);
      run_tests<dim, degree + 1, max_degree>(timings_table, memory_table);
    }
}

int
main()
{
  deallog << std::setprecision(10);
  ConvergenceTable timings_table;
  ConvergenceTable memory_table;

  run_tests<DEAL_DIMENSION, 1, 8>(timings_table, memory_table);
  timings_table.set_scientific("FEPointEvaluation", true);
  timings_table.set_scientific("FEEvaluation", true);
  timings_table.set_scientific("GhostPenalty", true);
  timings_table.set_scientific("SBMFace", true);
  timings_table.set_scientific("FullMatrix", true);
  timings_table.write_text(std::cout);

  std::cout << "\n Timings in microseconds per iteration, vectorization over:"
            << VectorizedArray<double>::size() << "\n " << std::endl;

  // memory_table.set_scientific("FEPointEvaluation", true);
  // memory_table.set_scientific("SBMFace", true);
  // memory_table.set_scientific("FullMatrix", true);
  memory_table.write_text(std::cout);
}