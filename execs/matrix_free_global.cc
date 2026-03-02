

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/cell_weights.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

#include "local_operators.h"
#include "matrix_free_operator.h"


const int dim      = DEAL_DIMENSION;
const int degree_u = 3;

bool TEST_VMULT_ONLY = true;

// #define PARALLEL_DISTRIBUTED

#ifdef PARALLEL_DISTRIBUTED
const constexpr bool PARALLEL = true;
#else
const constexpr bool PARALLEL = false;
#endif


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
    output += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count() /
              1000.;
  }

private:
  double &                                       output;
  std::chrono::high_resolution_clock::time_point start;
};


template <int dim>
class LevelSetFunction : public Function<dim>
{
public:
  LevelSetFunction(int n_balls)
    : Function<dim>(1)
    , radius((dim == 3 && n_balls != 1 ? 5. : 1.) * 1. / n_balls)
  {
    Assert(n_balls > 0, ExcInternalError());
    centers.resize(n_balls);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        for (auto &center : centers)
          for (unsigned int d = 0; d < dim; ++d)
            center(d) =
              2.0 * (std::rand() / static_cast<double>(RAND_MAX) - 0.5);
      }
    centers[0] = Point<dim>();
    MPI_Bcast(
      centers.data(), centers.size() * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // for (unsigned int i = 0;
    //      i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    //      ++i)
    //   {
    //     if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == i)
    //       {
    //         std::cout << "Rank " << i << ":\n";
    //         for (const auto &center : centers)
    //           std::cout << "Center: " << center << std::endl;
    //       }
    //     MPI_Barrier(MPI_COMM_WORLD);
    //   }
  }

  double
  value(const Point<dim> & point,
        const unsigned int component = 0) const override
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    double distance = point.norm();
    for (const auto &center : centers)
      distance = std::min(distance, (point - center).norm());
    return distance - radius;
  }

private:
  std::vector<Point<dim>> centers;
  const double            radius;
};

namespace cutFemMatrixFree
{
  using namespace dealii;

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  using LaplaceOperator =
    MatrixFreeOperator<LocalOperators::Laplace<dim, degree_u, double>,
                       VectorType>;
  using LevelOperatorType =
    MatrixFreeOperator<LocalOperators::Laplace<dim, degree_u, double>,
                       VectorType>;


  template <int dim>
  class LaplaceSolver
  {
  public:
    LaplaceSolver();

    void
    run();

  private:
    void
    make_grid();

    void
    setup_discrete_level_set(const unsigned int n_balls);

    void
    distribute_dofs();

    void
    initialize_matrices();

    double
    initialize_matrix_free();

    void
    assemble_system();

    void
    assemble_internal_matrix();

    double
    benchmark_vmult() const;

    void
    solve();

    std::pair<unsigned int, double>
    solve_mf();

    void
    output_results() const;

    double
    compute_L2_error() const;

    bool
    face_has_ghost_penalty(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const;

#ifdef PARALLEL_DISTRIBUTED
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    const unsigned int                     fe_degree;
    const Functions::ConstantFunction<dim> rhs_function;
    const Functions::ConstantFunction<dim> boundary_condition;


    std::unique_ptr<Function<dim>> level_set_function;
    const FE_Q<dim>                fe_level_set;
    DoFHandler<dim>                level_set_dof_handler;
    MappingCartesian<dim>          mapping;
    VectorType                     level_set;
    MGLevelObject<VectorType>      mg_level_set;

    hp::FECollection<dim>           fe_collection;
    DoFHandler<dim>                 dof_handler;
    const AffineConstraints<double> constraints;
    VectorType                      solution;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern                   sparsity_pattern;
    SparseMatrix<double>              stiffness_matrix;
    LaplaceOperator                   system_operator;
    MGLevelObject<LevelOperatorType>  level_operators;
    MGTransferMatrixFree<dim, double> mg_transfer;


    VectorType rhs;


    FullMatrix<double> internal_cell_matrix;

    const double penalty_parameter;
    bool         skip_assembly;

    unsigned int n_intersected_cells;
    unsigned int n_active_cells;

    MPI_Comm           mpi_global_communicator;
    MPI_Comm           mpi_local_communicator;
    ConditionalOStream pcout;
  };



  template <int dim>
  LaplaceSolver<dim>::LaplaceSolver()
    :
#ifdef PARALLEL_DISTRIBUTED
    triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
#else
    triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe_degree(degree_u)
    , rhs_function(4.0)
    , boundary_condition(0.0)
    , fe_level_set(fe_degree)
    , level_set_dof_handler(triangulation)
    , dof_handler(triangulation)
    , mesh_classifier(level_set_dof_handler, level_set)
    , penalty_parameter(30)
    , skip_assembly(true)
    , mpi_global_communicator(MPI_COMM_WORLD)
    , mpi_local_communicator(PARALLEL ? MPI_COMM_WORLD : MPI_COMM_SELF)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}



  template <int dim>
  void
  LaplaceSolver<dim>::make_grid()
  {
    pcout << "Creating background mesh" << std::endl;

    // GridGenerator::hyper_cube(triangulation, -1.21, 1.21);
    GridGenerator::subdivided_hyper_cube(triangulation, 2, -1.05, 1.05);
    // triangulation.refine_global(2);
  }



  template <int dim>
  void
  LaplaceSolver<dim>::setup_discrete_level_set(const unsigned int n_balls)
  {
    pcout << "Setting up discrete level set function with " << n_balls
          << " balls" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set_dof_handler.distribute_mg_dofs();
    // level_set.reinit(level_set_dof_handler.n_dofs());
    level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(
                       level_set_dof_handler),
                     mpi_local_communicator);


    // level_set_function.reset(new FunctionParser<dim>("(x^2 + y^2)^0.5 - 1"));
    level_set_function.reset(new LevelSetFunction<dim>(n_balls));
    // level_set_function.reset(
    //   new FunctionParser<dim>("(x^10 + y^10)^0.1 -    1"));

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             *level_set_function,
                             //  signed_distance_sphere,
                             level_set);
    level_set.update_ghost_values();
    pcout << "Classifying cells" << std::endl;
    mesh_classifier.reclassify();


    if constexpr (PARALLEL)
      {
        auto compute_weight =
          [&](const typename DoFHandler<dim>::cell_iterator &cell,
              const FiniteElement<dim> &) -> unsigned int {
          const NonMatching::LocationToLevelSet cell_location =
            mesh_classifier.location_to_level_set(cell);

          unsigned int result = 1;
          if (cell_location == NonMatching::LocationToLevelSet::intersected)
            result = std::pow(fe_degree, dim + 2);
          else if (cell_location == NonMatching::LocationToLevelSet::outside)
            result = 0;

          return static_cast<unsigned int>(result);
        };

        parallel::CellWeights<dim> cell_balancing(level_set_dof_handler,
                                                  compute_weight);

        triangulation.repartition();


        level_set_dof_handler.distribute_dofs(fe_level_set);
        level_set_dof_handler.distribute_mg_dofs();
        level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(
                           level_set_dof_handler),
                         mpi_local_communicator);

        VectorTools::interpolate(level_set_dof_handler,
                                 *level_set_function,
                                 level_set);

        level_set.update_ghost_values();
        pcout << "Classifying cells" << std::endl;
        mesh_classifier.reclassify();
      }
  }


  enum ActiveFEIndex
  {
    lagrange    = 0,
    intersected = 1,
    nothing     = 2,
  };

  template <int dim>
  void
  LaplaceSolver<dim>::distribute_dofs()
  {
    pcout << "Distributing degrees of freedom" << std::endl;
    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    unsigned int n_local_active_cells      = 0;
    unsigned int n_local_intersected_cells = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(ActiveFEIndex::nothing);
        else if (cell_location == NonMatching::LocationToLevelSet::intersected)
          {
            cell->set_active_fe_index(ActiveFEIndex::intersected);
            n_local_intersected_cells++;
            n_local_active_cells++;
          }
        else
          {
            Assert(cell_location == NonMatching::LocationToLevelSet::inside,
                   ExcInternalError());
            cell->set_active_fe_index(ActiveFEIndex::lagrange);
            n_local_active_cells++;
          }
      }
    n_active_cells = 0;
    MPI_Allreduce(&n_local_active_cells,
                  &n_active_cells,
                  1,
                  MPI_UNSIGNED,
                  MPI_SUM,
                  mpi_local_communicator);

    n_intersected_cells = 0;
    MPI_Allreduce(&n_local_intersected_cells,
                  &n_intersected_cells,
                  1,
                  MPI_UNSIGNED,
                  MPI_SUM,
                  mpi_local_communicator);

    dof_handler.distribute_dofs(fe_collection);
    return;
  }



  template <int dim>
  void
  LaplaceSolver<dim>::initialize_matrices()
  {
    pcout << "Initializing matrices" << std::endl;

    const auto face_has_flux_coupling = [&](const auto &       cell,
                                            const unsigned int face_index) {
      return this->face_has_ghost_penalty(cell, face_index);
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;


    const bool keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
  }


  template <int dim>
  double
  LaplaceSolver<dim>::initialize_matrix_free()
  {
    pcout << "Initializing matrix-free objects" << std::endl;
    double mf_init_time = 0;


    const auto initialize_level = [&]() {
      Timer timer(mf_init_time);

      typename MatrixFree<dim, double>::AdditionalData additional_data;


      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values);
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values);
      additional_data.store_ghost_cells = true;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      // additional_data.mg_level          = nlevels - 1;

      std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
        new MatrixFree<dim, double>());

      system_mf_storage->reinit(mapping,
                                dof_handler,
                                constraints,
                                QGauss<1>(fe_degree + 1),
                                additional_data);

      system_operator.initialize(system_mf_storage);



      system_operator.initialize_local_operator(penalty_parameter);
      system_operator.initialize_ghost_penalty_faces();
      system_operator.initilize_cut_cells_quadrature(
        QGauss<1>(fe_degree + 1),
        level_set_dof_handler,
        level_set,
        mapping,
        update_gradients | update_JxW_values,
        update_default,
        update_gradients | update_values | update_JxW_values |
          update_normal_vectors);
    };


    initialize_level();
    system_operator.initialize_dof_vector(solution);
    system_operator.initialize_dof_vector(rhs);

    return mf_init_time / 1e6; // Convert to seconds
  }


  template <int dim>
  bool
  LaplaceSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  }



  template <int dim>
  void
  LaplaceSolver<dim>::assemble_system()
  {
    pcout << "Assembling" << std::endl;

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double ghost_parameter = 1.;
    const double nitsche_parameter =
      penalty_parameter * (fe_degree + 1) * fe_degree;


    const QGauss<dim - 1>  face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                                 update_JxW_values |
                                                 update_normal_vectors |
                                                 update_hessians |
                                                 update_3rd_derivatives);


    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);


    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        if (mesh_classifier.location_to_level_set(cell) ==
            NonMatching::LocationToLevelSet::outside)
          continue;

        local_stiffness = 0;
        local_rhs       = 0;

        Quadrature<dim> indide_quadrature;

        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();


        if (inside_fe_values)
          indide_quadrature = inside_fe_values->get_quadrature();

        if (inside_fe_values)
          for (const unsigned int q :
               inside_fe_values->quadrature_point_indices())
            {
              const Point<dim> &point = inside_fe_values->quadrature_point(q);
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                  local_rhs(i) += rhs_function.value(point) *
                                  inside_fe_values->shape_value(i, q) *
                                  inside_fe_values->JxW(q);
                }
            }

        else
          {
            // std::cout << "no quadrature" << std::endl;
          }

        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          (-normal * surface_fe_values->shape_grad(i, q) *
                             surface_fe_values->shape_value(j, q) +
                           -normal * surface_fe_values->shape_grad(j, q) *
                             surface_fe_values->shape_value(i, q) +
                           nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) *
                             surface_fe_values->shape_value(j, q)) *
                          surface_fe_values->JxW(q);
                      }
                    local_rhs(i) +=
                      boundary_condition.value(point) *
                      (nitsche_parameter / cell_side_length *
                         surface_fe_values->shape_value(i, q) -
                       normal * surface_fe_values->shape_grad(i, q)) *
                      surface_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        if (!skip_assembly)
          stiffness_matrix.add(local_dof_indices, local_stiffness);

        rhs.add(local_dof_indices, local_rhs);

        for (const unsigned int f : cell->face_indices())
          if (face_has_ghost_penalty(cell, f))
            {
              // fixme: skip ghost penalty in MPI run
              if (skip_assembly)
                continue;

              const unsigned int invalid_subface =
                numbers::invalid_unsigned_int;


              fe_interface_values.reinit(cell,
                                         f,
                                         invalid_subface,
                                         cell->neighbor(f),
                                         cell->neighbor_of_neighbor(f),
                                         invalid_subface);

              const unsigned int n_interface_dofs =
                fe_interface_values.n_current_interface_dofs();
              FullMatrix<double> local_stabilization(n_interface_dofs,
                                                     n_interface_dofs);
              for (unsigned int q = 0;
                   q < fe_interface_values.n_quadrature_points;
                   ++q)
                {
                  const Tensor<1, dim> normal = fe_interface_values.normal(q);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    for (unsigned int j = 0; j < n_interface_dofs; ++j)
                      {
                        local_stabilization(i, j) +=
                          .5 * ghost_parameter *
                          (cell_side_length * normal *
                             fe_interface_values.jump_in_shape_gradients(i, q) *
                             normal *
                             fe_interface_values.jump_in_shape_gradients(j, q) +

                           std::pow(cell_side_length, 2 * 2 - 1) * 0.25 *
                             scalar_product(
                               normal *
                                 fe_interface_values.jump_in_shape_hessians(i,
                                                                            q),
                               normal * fe_interface_values
                                          .jump_in_shape_hessians(j, q)) +

                           std::pow(cell_side_length, 2 * 3 - 1) * (1. / 25.) *
                             scalar_product(
                               normal * fe_interface_values
                                          .jump_in_shape_3rd_derivatives(i, q),
                               normal * fe_interface_values
                                          .jump_in_shape_3rd_derivatives(j, q))

                             ) *
                          fe_interface_values.JxW(q);
                      }
                }

              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();


              stiffness_matrix.add(local_interface_dof_indices,
                                   local_stabilization);
            }
      }
    rhs.compress(VectorOperation::add);
    rhs.update_ghost_values();
  }

  template <int dim>
  void
  LaplaceSolver<dim>::solve()
  {
    pcout << "Solving system" << std::endl;



    const unsigned int   max_iterations = 3 * solution.size();
    SolverControl        solver_control(max_iterations, 1e-8);
    SolverCG<VectorType> solver(solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize(stiffness_matrix);

    solver.solve(system_operator, solution, rhs, preconditioner);

    pcout << "Solved in " << solver_control.last_step() << " iterations"
          << std::endl;
  }


  template <int dim>
  std::pair<unsigned int, double>
  LaplaceSolver<dim>::solve_mf()
  {
    pcout << "Solving system with matrix-free" << std::endl;
    // const unsigned int max_level = triangulation.n_global_levels() - 1;
    // using StorageType = PatchOperator::PatchStorage<MatrixFree<dim, double>>;
    // using PatchSmootherType =
    //   PatchOperator::LaplaceSmoother<LevelOperatorType, double>;
    // mg::SmootherRelaxation<PatchSmootherType, VectorType> mg_smoother;

    IterationNumberControl  solver_control(400000, 1.e-8);
    SolverGMRES<VectorType> solver(solver_control);

    // TrilinosWrappers::PreconditionAMG preconditionerAMG;
    // preconditionerAMG.initialize(stiffness_matrix);


    solver.solve(system_operator, solution, rhs, PreconditionIdentity());

    pcout << "Solved in " << solver_control.last_step() << " iterations"
          << " with residual " << solver_control.last_value() << std::endl;

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  double
  LaplaceSolver<dim>::benchmark_vmult() const
  {
    pcout << "Benchmarking vmult" << std::endl;

    const unsigned int n_repetitions = 100;
    VectorType         tmp(solution);


    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < n_repetitions; ++i)
      system_operator.vmult(tmp, rhs);
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    pcout << "Thoughput: "
          << dof_handler.n_dofs() /
               (duration / static_cast<double>(n_repetitions))
          << std::endl;

    return duration / static_cast<double>(n_repetitions);
  }


  template <int dim>
  void
  LaplaceSolver<dim>::output_results() const
  {
    pcout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    const auto data = system_operator.get_matrix_free();

    // Vector<float> cell_mf_index(triangulation.n_active_cells());
    // for (unsigned int cell_b = 0; cell_b < data->n_cell_batches(); ++cell_b)
    //   for (unsigned int lane = 0;
    //        lane < data->n_active_entries_per_cell_batch(cell_b);
    //        ++lane)
    //     {
    //       auto cell = data->get_cell_iterator(cell_b, lane);
    //       cell_mf_index(cell->index()) =
    //         cell_b * VectorizedArray<double>::size() + lane;
    //     }
    // data_out.add_data_vector(cell_mf_index, "MF_index");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(fe_degree);

    // std::ofstream output("matrix_free-" +
    //                      std::to_string(triangulation.n_global_levels()) +
    //                      ".vtu");
    // data_out.write_vtu(output);
    if (PARALLEL)
      data_out.write_vtu_with_pvtu_record("results/",
                                          "matrix_free",
                                          triangulation.n_global_levels(),
                                          mpi_global_communicator);
    else if (Utilities::MPI::this_mpi_process(mpi_global_communicator) == 0)
      data_out.write_vtu_with_pvtu_record("results/",
                                          "matrix_free",
                                          triangulation.n_global_levels(),
                                          mpi_global_communicator);
  }



  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double
    value(const Point<dim> & point,
          const unsigned int component = 0) const override;
  };



  template <int dim>
  double
  AnalyticalSolution<dim>::value(const Point<dim> & point,
                                 const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    return 0. - 2. / dim * (point.norm_square() - 1.);
  }



  template <int dim>
  double
  LaplaceSolver<dim>::compute_L2_error() const
  {
    std::cout << "Computing L2 error" << std::endl;

    solution.update_ghost_values();

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    AnalyticalSolution<dim> analytical_solution;
    double                  error_L2_squared = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (mesh_classifier.location_to_level_set(cell) ==
            NonMatching::LocationToLevelSet::outside)
          continue;
        if (!cell->is_locally_owned())
          continue;

        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(solution, solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * fe_values->JxW(q);
              }
          }
      }

    return std::sqrt(error_L2_squared);
  }



  template <int dim>
  void
  LaplaceSolver<dim>::run()
  {
    ConvergenceTable   convergence_table;
    const unsigned int min_refinements = TEST_VMULT_ONLY ? 4 : 0;
    const unsigned int max_refinements = 12;
    const unsigned int n_init_balls    = 1;

    make_grid();
    triangulation.refine_global(min_refinements);
    for (unsigned int cycle = min_refinements; cycle <= max_refinements;
         cycle++)
      for (unsigned int n_balls = n_init_balls; n_balls <= 25; n_balls++)
        {
          pcout << "Refinement cycle " << cycle << std::endl;
          if (cycle > 0 && n_balls == n_init_balls)
            triangulation.refine_global(1);
          pcout << "Number of active cells: " << triangulation.n_active_cells()
                << std::endl;
          setup_discrete_level_set(n_balls);


          distribute_dofs();
          if (!skip_assembly)
            initialize_matrices();
          auto mf_init_time = initialize_matrix_free();
          assemble_system();
          if (!TEST_VMULT_ONLY)
            {
              auto [iterations, residual] = solve_mf();
              convergence_table.add_value("Iter", iterations);
              convergence_table.add_value("Residual", residual);
              convergence_table.set_scientific("Residual", true);

              const double error_L2 = compute_L2_error();
              convergence_table.add_value("L2-Error", error_L2);
              convergence_table.evaluate_convergence_rates(
                "L2-Error", ConvergenceTable::reduction_rate_log2);
              convergence_table.set_scientific("L2-Error", true);
            }

          output_results();

          double       vmult_time = benchmark_vmult();
          auto         timings    = system_operator.get_timing(true);
          const double cell_side_length =
            triangulation.begin_active()->minimum_vertex_distance();

          convergence_table.add_value("Cycle", cycle);
          convergence_table.add_value("NBalls", n_balls);
          convergence_table.add_value("Mesh size", cell_side_length);
          convergence_table.add_value("DOFs", dof_handler.n_dofs());
          convergence_table.add_value("VMULT Time", vmult_time);
          convergence_table.set_scientific("VMULT Time", true);
          convergence_table.add_value("Interior T", timings[0]);
          convergence_table.set_scientific("Interior T", true);
          convergence_table.add_value("Cut T", timings[1]);
          convergence_table.set_scientific("Cut T", true);
          convergence_table.add_value("Ghost T", timings[2]);
          convergence_table.set_scientific("Ghost T", true);
          convergence_table.add_value("MPI", timings[3]);
          convergence_table.set_scientific("MPI", true);
          convergence_table.add_value("Total T", timings[4]);
          convergence_table.set_scientific("Total T", true);

          convergence_table.add_value("MF init T", mf_init_time);
          convergence_table.set_scientific("MF init T", true);

          convergence_table.add_value("Cells fraction",
                                      n_intersected_cells /
                                        static_cast<double>(n_active_cells));

          convergence_table.add_value(
            "MF Mem", system_operator.memory_consumption_geometry());



          pcout << std::endl;
          if (Utilities::MPI::this_mpi_process(mpi_global_communicator) == 0)
            convergence_table.write_text(std::cout);
          pcout << std::endl;
        }
  }

} // namespace cutFemMatrixFree



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);


  cutFemMatrixFree::LaplaceSolver<dim> laplace_solver;
  laplace_solver.run();
}