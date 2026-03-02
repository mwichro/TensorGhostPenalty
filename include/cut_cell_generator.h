#ifndef __CUT_CELL_GENERATOR_H__
#define __CUT_CELL_GENERATOR_H__


#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/immersed_surface_quadrature.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <string>
#include <vector>

#include "renumber_lexycogrhaphic.h"


namespace cutCellTools
{
  using namespace dealii;

  template <int dim>
  class Generator
  {
  public:
    Generator(const FiniteElement<dim> &level_set_fe,
              const Quadrature<1> &     quad_1D,
              const bool                lexycographic_order = false)
      : level_set_dof_handler(tria)
      , quadratures1D(quad_1D)
    {
      GridGenerator::hyper_cube(tria, 0, 1.);
      level_set_dof_handler.distribute_dofs(level_set_fe);
      level_set.reinit(level_set_dof_handler.n_dofs());

      if (lexycographic_order)
        DoFRenumbering::renumber_lexycographic(level_set_dof_handler);

      quadrature_generator =
        std::make_unique<NonMatching::DiscreteQuadratureGenerator<dim>>(
          quadratures1D, level_set_dof_handler, level_set);
    }



    void
    print() const;

    void
    output_gnuplot();

    void
    reinit(Vector<double> level_set_in)
    {
      level_set = level_set_in;

      // Check if level_set contains both non-negative and non-positive values
      bool has_non_negative = false;
      bool has_non_positive = false;
      for (const auto &value : level_set)
        {
          if (value >= 0)
            has_non_negative = true;
          if (value <= 0)
            has_non_positive = true;
        }
      // Assert(
      //   has_non_negative && has_non_positive,
      //   ExcMessage(
      //     "level_set must contain both non-negative and non-positive
      //     values."));
      quadrature_generator->generate(level_set_dof_handler.begin_active());
    }



    const auto &
    get_inside_quadrature() const
    {
      return quadrature_generator->get_inside_quadrature();
    }

    const auto &
    get_outside_quadrature() const
    {
      return quadrature_generator->get_outside_quadrature();
    }

    const auto &
    get_surface_quadrature() const
    {
      return quadrature_generator->get_surface_quadrature();
    }


  private:
    Triangulation<dim> tria;
    DoFHandler<dim>    level_set_dof_handler;
    Vector<double>     level_set;

    std::unique_ptr<NonMatching::DiscreteQuadratureGenerator<dim>>
                             quadrature_generator;
    const hp::QCollection<1> quadratures1D;

    Quadrature<dim>                             inside_quad;
    Quadrature<dim>                             outside_quad;
    NonMatching::ImmersedSurfaceQuadrature<dim> surface_quad;
  };



} // namespace cutCellTools
#endif // __CUT_CELL_GENERATOR_H__