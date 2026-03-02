#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace ClosestPoint
{
  using namespace dealii;

  template <int dim>
  double
  find_closest_surface_point(const Point<dim> &        point,
                             const FiniteElement<dim> &fe,
                             const std::vector<double> dof_values,
                             Point<dim> &              closest_point)
  {
    AssertDimension(dof_values.size(), fe.dofs_per_cell);

    const double tolerance = 1e-10;
    // X, Y, Z, lambda
    Vector<double> current_solution(dim + 1);
    Vector<double> solution_update(dim + 1);

    Vector<double>     residual(dim + 1);
    FullMatrix<double> hessian(dim + 1, dim + 1);

    for (unsigned int i = 0; i < dim; ++i)
      current_solution[i] = closest_point[i];


    for (unsigned int newton_iter = 0; newton_iter < 20; ++newton_iter)
      {
        hessian             = 0.0;
        residual            = 0.0;
        const double lambda = current_solution[dim];
        for (unsigned int k = 0; k < dof_values.size(); ++k)
          {
            const auto value_k = fe.shape_value(k, closest_point);
            const auto grad_k  = fe.shape_grad(k, closest_point);
            const auto hess_k  = fe.shape_grad_grad(k, closest_point);
            for (unsigned int i = 0; i < dim; ++i)
              {
                for (unsigned int j = 0; j < dim; ++j)
                  hessian(i, j) += lambda * dof_values[k] * hess_k[i][j];

                hessian(i, dim) += dof_values[k] * grad_k[i];
                hessian(dim, i) += dof_values[k] * grad_k[i];
              }

            for (unsigned int i = 0; i < dim; ++i)
              residual[i] -= lambda * dof_values[k] * grad_k[i];

            residual[dim] -= dof_values[k] * value_k;
          }


        for (unsigned int i = 0; i < dim; ++i)
          {
            residual[i] -= current_solution[i] - point[i];
            hessian[i][i] += 1.0;
          }

        if (residual.l2_norm() < tolerance)
          break;


        hessian.gauss_jordan();
        hessian.vmult(solution_update, residual);
        current_solution += solution_update;

        for (unsigned int i = 0; i < dim; ++i)
          closest_point[i] = current_solution[i];
      }

    Assert(residual.l2_norm() < 1e-10,
           dealii::ExcMessage("Newton iteration did not converge"));

    return residual.l2_norm();
  }

  template <int dim>
  std::vector<Point<dim>>
  shifted_boundary_points(const std::vector<Point<dim>> &quadrature_points,
                          const unsigned                 face_index,
                          const FiniteElement<dim> &     fe,
                          const std::vector<double>      dof_values)
  {
    // compute the shift
    Point<dim> shift;
    switch (face_index)
      {
        case 0:
          shift(0) = 1.0;
          break;
        case 1:
          shift(0) = -1.0;
          break;
        case 2:
          shift(1) = 1.0;
          break;
        case 3:
          shift(1) = -1.0;
          break;
        case 4:
          shift(2) = 1.0;
          break;
        case 5:
          shift(2) = -1.0;
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid face index for given dim"));
          break;
      }

    unsigned perpencular_coordinate = 0;
    if (face_index < 2)
      perpencular_coordinate = 0;
    else if (face_index < 4)
      perpencular_coordinate = 1;
    else
      perpencular_coordinate = 2;

    AssertThrow(perpencular_coordinate < dim,
                ExcMessage("Invalid face index for given dim"));


    // shift the quadrature points to neighbour cell
    std::vector<Point<dim>> shifted_points(quadrature_points.size());
    for (size_t i = 0; i < quadrature_points.size(); ++i)
      {
        shifted_points[i] = quadrature_points[i] + shift;
      }

    std::vector<Point<dim>> closest_points(shifted_points);

    // midway as a starting point
    // closest_points[0](perpencular_coordinate) = 0.5;
    for (size_t i = 0; i < shifted_points.size(); ++i)
      {
        Assert(std::fabs(quadrature_points[i][perpencular_coordinate]) < 1e-6 ||
                 std::fabs(quadrature_points[i][perpencular_coordinate] - 1.0) <
                   1e-6,
               ExcMessage(
                 "Invalid quadrature point for given face index " +
                 std::to_string(face_index) + " at point " + std::to_string(i) +
                 " coordinate " + std::to_string(perpencular_coordinate) +
                 " value " +
                 std::to_string(quadrature_points[i][perpencular_coordinate])));

        if (i != 0)
          closest_points[i](perpencular_coordinate) =
            closest_points[i - 1](perpencular_coordinate);


        find_closest_surface_point(shifted_points[i],
                                   fe,
                                   dof_values,
                                   closest_points[i]);
      }

    // back to original cell
    for (size_t i = 0; i < shifted_points.size(); ++i)
      closest_points[i] -= shift;

    return closest_points;
  }

  template <int dim, class VECTOR>
  class ShiftsGenerator
  {
  public:
    using VectorType = VECTOR;
    ShiftsGenerator(const VectorType &     level_set,
                    const DoFHandler<dim> &dof_handler,
                    const unsigned int     level);

    std::pair<std::vector<Point<dim>>, std::vector<Point<dim>>>
    generate_shifts(
      const typename Triangulation<dim>::cell_iterator &search_cell,
      const typename Triangulation<dim>::cell_iterator &reference_cell,
      const std::vector<Point<dim>> &quadrature_points) const;

  private:
    const unsigned int     level;
    const DoFHandler<dim> &dof_handler;
    const VectorType &     level_set;
    MappingCartesian<dim>  mapping;
  };

  template <int dim, class VECTOR>
  ShiftsGenerator<dim, VECTOR>::ShiftsGenerator(
    const VectorType &     level_set,
    const DoFHandler<dim> &dof_handler,
    const unsigned int     level)
    : level(level)
    , dof_handler(dof_handler)
    , level_set(level_set)
  {
    if (level != numbers::invalid_unsigned_int)
      {
        AssertThrow(level < dof_handler.get_triangulation().n_global_levels(),
                    dealii::ExcMessage("Level is larger than number of levels "
                                       "in the triangulation"));
      }
  }

  template <int dim, class VECTOR>
  std::pair<std::vector<Point<dim>>, std::vector<Point<dim>>>
  ShiftsGenerator<dim, VECTOR>::generate_shifts(
    const typename Triangulation<dim>::cell_iterator &search_cell,
    const typename Triangulation<dim>::cell_iterator &reference_cell,
    const std::vector<Point<dim>> &                   quadrature_points) const
  {
    std::vector<Point<dim>> closest_unit_search_points(quadrature_points);
    for (unsigned int q = 0; q < quadrature_points.size(); ++q)
      closest_unit_search_points[q] =
        mapping.transform_real_to_unit_cell(search_cell, quadrature_points[q]);

    std::vector<double> dof_values_level_set(
      dof_handler.get_fe().dofs_per_cell);
    std::vector<types::global_dof_index> level_set_dof_indices(
      dof_handler.get_fe().dofs_per_cell);

    typename DoFHandler<dim>::cell_iterator dof_cell(
      &search_cell->get_triangulation(),
      search_cell->level(),
      search_cell->index(),
      &dof_handler);

    if (level != numbers::invalid_unsigned_int)
      dof_cell->get_mg_dof_indices(level_set_dof_indices);
    else
      dof_cell->get_dof_indices(level_set_dof_indices);

    level_set.extract_subvector_to(level_set_dof_indices.begin(),
                                   level_set_dof_indices.end(),
                                   dof_values_level_set.begin());

    for (size_t i = 0; i < closest_unit_search_points.size(); ++i)
      {
        find_closest_surface_point(closest_unit_search_points[i],
                                   dof_handler.get_fe(),
                                   dof_values_level_set,
                                   closest_unit_search_points[i]);
      }
    std::vector<Point<dim>> closest_real_points(quadrature_points.size());
    std::vector<Point<dim>> closest_unit_reference_points(
      quadrature_points.size());
    // back to absolute coordinates
    for (unsigned int q = 0; q < quadrature_points.size(); ++q)
      closest_real_points[q] =
        mapping.transform_unit_to_real_cell(search_cell,
                                            closest_unit_search_points[q]);

    for (unsigned int q = 0; q < quadrature_points.size(); ++q)
      closest_unit_reference_points[q] =
        mapping.transform_real_to_unit_cell(reference_cell,
                                            closest_real_points[q]);

    return {closest_real_points, closest_unit_reference_points};
  }


  template <int dim>
  void
  output_shifts(const std::string &filename_without_extension,
                const std::vector<std::pair<Point<dim>, Point<dim>>> &shifts,
                const unsigned int n_mpi_process,
                const unsigned int this_mpi_process)
  {
    using PointOutData = DataOutBase::Patch<1, dim>;
    std::vector<PointOutData> shifts_out;
    shifts_out.reserve(shifts.size());

    std::vector<std::string> data_names;
    // data_names.emplace_back("index");


    // const unsigned n_datasets = 1;
    Triangulation<1, dim> tria_dummy;
    GridGenerator::hyper_cube(tria_dummy, -1., 1.);

    for (unsigned int shift_index = 0; shift_index < shifts.size();
         ++shift_index)
      {
        PointOutData link_out;
        link_out.patch_index    = shift_index;
        link_out.reference_cell = tria_dummy.begin_active()->reference_cell();

        link_out.vertices[0] = shifts[shift_index].first;
        link_out.vertices[1] = shifts[shift_index].second;
        shifts_out.push_back(link_out);
      }

    std::vector<std::string> piece_names(n_mpi_process);
    for (unsigned int i = 0; i < n_mpi_process; ++i)
      piece_names[i] = filename_without_extension + ".proc" +
                       Utilities::int_to_string(i, 4) + ".vtu";
    std::string new_file = piece_names[this_mpi_process];

    std::string out_pvtu = filename_without_extension + ".pvtu";


    std::ofstream out(new_file);
    std::vector<
      std::tuple<unsigned int,
                 unsigned int,
                 std::string,
                 DataComponentInterpretation::DataComponentInterpretation>>
      vector_data_ranges;

    DataOutBase::VtkFlags vtu_flags;

    DataOutBase::write_vtu(
      shifts_out, data_names, vector_data_ranges, vtu_flags, out);
    if (this_mpi_process == 0)
      {
        std::ofstream pvtu_output(out_pvtu);
        std::ostream &pvtu_out_steam = pvtu_output;
        DataOutBase::write_pvtu_record(pvtu_out_steam,
                                       piece_names,
                                       data_names,
                                       vector_data_ranges,
                                       vtu_flags);
      }
  }


} // namespace ClosestPoint
