#ifndef __RENUMBER_LEXYCOGRHAPHIC_H__
#define __RENUMBER_LEXYCOGRHAPHIC_H__



#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/full_matrix.h>

#include <fstream>
#include <vector>
namespace dealii::DoFRenumbering
{
  using namespace dealii;

  template <int dim>
  void
  renumber_lexycographic(DoFHandler<dim> &handler)
  {
    std::map<types::global_dof_index, Point<dim>> dof_location_map;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                         handler,
                                         dof_location_map);
    std::vector<std::pair<types::global_dof_index, Point<dim>>>
      dof_location_vector;

    for (const auto &s : dof_location_map)
      dof_location_vector.push_back(s);

    std::sort(
      dof_location_vector.begin(),
      dof_location_vector.end(),
      [&](const std::pair<types::global_dof_index, Point<dim>> &p1,
          const std::pair<types::global_dof_index, Point<dim>> &p2) -> bool {
        for (int i = dim - 1; i >= 0; --i)
          {
            if (p1.second(i) < p2.second(i))
              return true;
            else if (p1.second(i) > p2.second(i))
              return false;
          }
        return false; // Equal points
      });


    std::vector<types::global_dof_index> renumering(dof_location_vector.size(),
                                                    0);
    for (types::global_dof_index dof = 0; dof < dof_location_vector.size();
         ++dof)
      renumering[dof_location_vector[dof].first] = dof;


    handler.renumber_dofs(renumering);
  }


} // namespace dealii::DoFRenumbering

#endif // __RENUMBER_LEXYCOGRHAPHIC_H__