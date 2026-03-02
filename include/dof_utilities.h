#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <vector>

#include "renumber_lexycogrhaphic.h"


namespace DoFUtilities
{
  using namespace dealii;

  template <int dim>
  std::vector<types::global_dof_index>
  extract_component_dofs(
    std::vector<types::global_dof_index> &local_dof_indices,
    const FiniteElement<dim> &            fe,
    const unsigned int                    component)
  {
    AssertDimension(local_dof_indices.size(), fe.dofs_per_cell);

    const auto base_index = fe.component_to_base_index(component).first;


    std::vector<types::global_dof_index> local_component_dof_indices(
      fe.base_element(base_index).dofs_per_cell, numbers::invalid_dof_index);

    for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
      if (fe.system_to_component_index(i).first == component)
        local_component_dof_indices[fe.system_to_component_index(i).second] =
          local_dof_indices[i];


    // Assert that there are no invalid_dof_index in local_indices
    for (const auto &index : local_component_dof_indices)
      Assert(index != numbers::invalid_dof_index,
             ExcMessage("Invalid DoF index found in local_indices_component"));

    return local_component_dof_indices;
  }

  template <int dim>
  class NumberingGenerator
  {
  public:
    NumberingGenerator(const unsigned int degree, const unsigned int direction)
      : degree(degree)
      , direction(direction)
      , dof_handler(tria)
      , fe(degree)
    {
      AssertIndexRange(direction, dim + 1);
      make_grid();
      intinialize_dofs();
    }

    const unsigned int &
    n_interface_dofs()
    {
      return n_dofs;
    }

    const auto &
    get_cell_tensor2face_tensor_numbering() const
    {
      return cell_tensor2face_tensor_numbering;
    }

    const auto &
    get_cell2tensor_numbering() const
    {
      return cell2tensor_numbering;
    }

    std::vector<unsigned int>
    get_duplicated_dofs() const;


    void
    output_dofs_location() const
    {
      const std::string filename = "dof_locations_dim" + std::to_string(dim) +
                                   "_dir" + std::to_string(direction) +
                                   ".gnuplot";

      std::map<types::global_dof_index, Point<dim>> dof_location_map =
        DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler);

      std::ofstream dof_location_file(filename);
      DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                     dof_location_map);

      auto cell = dof_handler.begin_active();
    }


  private:
    void
    make_grid()
    {
      Point<dim> second;
      for (unsigned int d = 0; d < dim; ++d)
        second(d) = 1;
      second(direction) = 2;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[direction] = 2;

      GridGenerator::subdivided_hyper_rectangle(tria,
                                                subdivisions,
                                                Point<dim>(),
                                                second);
      if (dim == 2 && direction == 1)
        GridTools::rotate(-M_PI / 2., tria);
      else if constexpr (dim == 3)
        {
          if (direction == 1)
            GridTools::rotate(Tensor<1, 3, double>({0, 0, 1}),
                              -M_PI / 2.,
                              tria);
          // FIXME!
          else if (direction == 2)
            GridTools::rotate(Tensor<1, 3, double>({0, 1, 0}), M_PI / 2., tria);
        }
    }

    void
    intinialize_dofs()
    {
      dof_handler.distribute_dofs(fe);
      DoFRenumbering::renumber_lexycographic(dof_handler);

      n_dofs                           = dof_handler.n_dofs();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      cell2tensor_numbering[0].resize(dofs_per_cell);
      cell2tensor_numbering[1].resize(dofs_per_cell);

      auto cell = dof_handler.begin_active();
      cell->get_dof_indices(cell2tensor_numbering[0]);
      for (unsigned int f = 0; f < dim * 2; ++f)
        if (!cell->at_boundary(f))
          {
            face_index = f;
            break;
          }
      ++cell;
      cell->get_dof_indices(cell2tensor_numbering[1]);

      const std::vector<unsigned int> dof_numbering =
        FEEvaluation<dim, -1, 1, 1, double>(fe, QGauss<1>(1), update_default)
          .get_internal_dof_numbering();

      const auto permuatate_dofs = [&](
        const std::vector<types::global_dof_index> &dofs) -> auto
      {
        auto result = dofs;
        for (unsigned int i = 0; i < dofs.size(); ++i)
          result[i] = dofs[dof_numbering[i]];
        return result;
      };

      for (unsigned int c = 0; c < 2; ++c)
        cell_tensor2face_tensor_numbering[c] =
          permuatate_dofs(cell2tensor_numbering[c]);
    }


    const unsigned int degree;
    const unsigned int direction;
    unsigned int       face_index;

    Triangulation<dim> tria;
    DoFHandler<dim>    dof_handler;
    FE_Q<dim>          fe;

    unsigned int                         n_dofs;
    std::vector<types::global_dof_index> interfece_to_tensor_product_numbering;
    std::array<std::vector<types::global_dof_index>, 2> cell2tensor_numbering;
    std::array<std::vector<types::global_dof_index>, 2>
      cell_tensor2face_tensor_numbering;
  };



  template <int dim>
  class InterfaceReindexer
  {
  public:
    InterfaceReindexer(const unsigned int &degree)
      : degree(degree)
    {
      for (unsigned int direction = 0; direction < dim; ++direction)
        {
          NumberingGenerator<dim> numbering_generator(degree, direction);
          cell_dof_to_interface_tensor[direction] =
            numbering_generator.get_cell2tensor_numbering();

          if (direction == 0)
            n_interface_dofs = numbering_generator.n_interface_dofs();
          else
            Assert(n_interface_dofs == numbering_generator.n_interface_dofs(),
                   ExcMessage("Inconsistent number of interface DoFs"));
        }
      if constexpr (dim == 1)
        {
          face_indices_ = {{1}};
        }
      if constexpr (dim == 2)
        {
          face_indices_ = {{1, 3}};
        }
      if constexpr (dim == 3)
        {
          face_indices_ = {{1, 3, 5}};
        }
    }

    std::vector<types::global_dof_index>
    reindex(const std::vector<types::global_dof_index> &local_indices,
            const std::vector<types::global_dof_index> &neigbor_dof_indices,
            const unsigned int                          face) const
    {
      Assert(face % 2 == 1 && face < 2 * dim, // Faces: 1, 3, 5
             ExcMessage("Invalid face index in InterfaceReindexer"));

      const unsigned int direction = face / 2;

      const auto &cell2interface_numbering =
        cell_dof_to_interface_tensor[direction];


      std::vector<unsigned int> interface_dof_indices(
        n_interface_dofs, numbers::invalid_dof_index);


      for (unsigned int i = 0; i < local_indices.size(); ++i)
        interface_dof_indices[cell2interface_numbering[0][i]] =
          local_indices[i];

      for (unsigned int i = 0; i < neigbor_dof_indices.size(); ++i)
        {
          const unsigned int index_i = cell2interface_numbering[1][i];
          if (interface_dof_indices[index_i] != numbers::invalid_dof_index)
            AssertDimension(interface_dof_indices[index_i],
                            neigbor_dof_indices[i]);

          interface_dof_indices[index_i] = neigbor_dof_indices[i];
        }

      // Assert every entry has been filled:
      for (const auto &dof_index : interface_dof_indices)
        Assert(dof_index != numbers::invalid_dof_index,
               ExcMessage("Invalid DoF index found in interface_dof_indices"));

      return interface_dof_indices;
    }

    const auto &
    face_indices() const
    {
      return face_indices_;
    }

  private:
    const unsigned int   degree;
    std::array<int, dim> face_indices_;
    unsigned int         n_interface_dofs;
    std::array<std::array<std::vector<unsigned int>, 2>, dim>
      cell_dof_to_interface_tensor;
  };


} // namespace DoFUtilities