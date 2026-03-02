#ifndef __GHOST_PENALTY_OPERATOR_H__
#define __GHOST_PENALTY_OPERATOR_H__


#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <string>
#include <vector>

#include "cut_cell_generator.h"
#include "renumber_lexycogrhaphic.h"


namespace GhostPenalty
{
  using namespace dealii;

  template <class ContIN, class ContOut>
  void
  inverse_permutation_n(const ContIN &first, ContOut &d_first)
  {
    for (std::size_t i = 0; i < first.size(); ++i)
      {
        d_first[first[i]] = i;
      }
  }



  template <int dim, typename Number>
  class TensorProductApplier;

  template <int dim>
  class Generator
  {
  public:
    Generator(const unsigned int degree, const unsigned int direction)
      : degree(degree)
      , direction(direction)
      , dof_handler(tria)
      , fe(degree)
      , fe_1d(degree)
    {
      AssertIndexRange(direction, dim + 1);
      initialize();
    }


    const auto &
    get_cell_tensor2face_tensor_numbering() const
    {
      return cell_tensor2face_tensor_numbering;
    }

    const auto &
    get_cell2tensor_numberig() const
    {
      return cell2tensor_numbering;
    }

    std::vector<unsigned int>
    get_duplicated_dofs() const;

    void
    print() const;
    void
    output_gnuplot();

    const unsigned int &
    n_interface_dofs() const
    {
      return n_dofs;
    }

    const auto &
    get_1D_ghost_penalty() const
    {
      return penalty_matrix;
    }

    const auto &
    get_1D_mass() const
    {
      return mass_matrix;
    }

    const auto &
    get_dim_ghost_penalty() const
    {
      return ghost_penalty_matrix;
    }

    const auto &
    get_face() const
    {
      return face_index;
    }

    template <int, typename>
    friend class TensorProductApplier;

  private:
    void
    initialize();


    void
    make_grid_and_dofs();



    void
    assemble_interface_matrix();

    void
    assemble_1D_mass_matrix();

    void
    assemble_1D_interface_matrix();

    const unsigned int degree;
    const unsigned int direction;
    Triangulation<dim> tria;
    DoFHandler<dim>    dof_handler;
    FE_Q<dim>          fe;
    FE_Q<1>            fe_1d;
    FullMatrix<double> ghost_penalty_matrix;
    FullMatrix<double> mass_matrix;
    FullMatrix<double> penalty_matrix;
    unsigned int       face_index;


    unsigned int                         n_dofs;
    std::vector<types::global_dof_index> interfece_to_tensor_product_numbering;
    std::array<std::vector<types::global_dof_index>, 2> cell2tensor_numbering;
    std::array<std::vector<types::global_dof_index>, 2>
      cell_tensor2face_tensor_numbering;
  };

  class Generator1D
  {
  public:
    Generator1D(unsigned int degree)
      : degree(degree)
    {
      polynomial_basis.resize(degree + 1);

      auto support_points = FE_Q<1>(degree).get_unit_support_points();
      std::sort(support_points.begin(),
                support_points.end(),
                [](const Point<1> &p, const Point<1> &q) -> bool {
                  return p(0) < q(0);
                });

      polynomial_basis[0] =
        Polynomials::generate_complete_Lagrange_basis(support_points);
      //   Polynomials::LagrangeEquidistant::generate_complete_basis(degree);


      for (unsigned int k = 1; k < degree + 1; ++k)
        {
          polynomial_basis[k].reserve(degree + 1);
          for (unsigned int i = 0; i < degree + 1; ++i)
            polynomial_basis[k].push_back(
              polynomial_basis[k - 1][i].derivative());
        }

      penalty_matrices.resize(degree);

      assemble_mass_matrix();
      assemble_laplace_matrix();
      for (unsigned int k = 1; k < degree + 1; ++k)
        assemble_penalty_matrix(k);
    }

    FullMatrix<double>
    get_mass_matrix(const double &h)
    {
      FullMatrix<double> scaled_mass = mass_matrix;
      scaled_mass *= h;
      return scaled_mass;
    }

    FullMatrix<double>
    get_patch_mass_matrix(const double &h)
    {
      FullMatrix<double> scaled_mass;
      scaled_mass.copy_from(get_mass_matrix(h));
      const unsigned int N = scaled_mass.m();
      FullMatrix<double> patch_mass(N * 2 - 1, N * 2 - 1);

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            patch_mass(i, j) += scaled_mass(i, j);
            patch_mass(i + N - 1, j + N - 1) += scaled_mass(i, j);
          }
      return patch_mass;
    }

    FullMatrix<double>
    get_penalty_matrix(const double &h)
    {
      FullMatrix<double> penalty = penalty_matrices[0];
      // Scaling:
      // From integration derivative h^-k * h^-k =h^(-2k)
      // in the form: h^(2k-1)
      // leaving  (1/h) * 1/(k!)^2

      double inverse_factorial_square = 1.;
      for (unsigned int k = 2; k <= degree; ++k)
        {
          inverse_factorial_square /= (k * k);
          penalty.add(inverse_factorial_square, penalty_matrices[k - 1]);
        }
      penalty *= (1 / h);
      return penalty;
    }

    FullMatrix<double>
    get_laplace_matrix(const double h) const
    {
      FullMatrix<double> scaled_laplace = laplace_matrix;
      scaled_laplace *= 1. / h;
      return scaled_laplace;
    }

    void
    print()
    {
      std::cout << "\n MASS \n";
      mass_matrix.print(std::cout, 10, 3);
      std::cout << "\n PENALTY \n";
      for (const auto &penalty_matrix : penalty_matrices)
        {
          penalty_matrix.print(std::cout, 10, 3);
          std::cout << std::endl;
        }
    }

  private:
    const unsigned int                                        degree;
    std::vector<std::vector<Polynomials::Polynomial<double>>> polynomial_basis;
    std::vector<FullMatrix<double>>                           penalty_matrices;
    FullMatrix<double>                                        mass_matrix;
    FullMatrix<double>                                        laplace_matrix;

    void
    assemble_mass_matrix()
    {
      const unsigned int N = degree + 1;
      mass_matrix.reinit(N, N);

      QGauss<1> quadrature(N);

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            mass_matrix(i, j) +=
              polynomial_basis[0][i].value(quadrature.point(q)(0)) *
              polynomial_basis[0][j].value(quadrature.point(q)(0)) *
              quadrature.weight(q);
    }

    void
    assemble_laplace_matrix()
    {
      const unsigned int N = degree + 1;
      laplace_matrix.reinit(N, N);

      QGauss<1> quadrature(N);

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            laplace_matrix(i, j) +=
              polynomial_basis[1][i].value(quadrature.point(q)(0)) *
              polynomial_basis[1][j].value(quadrature.point(q)(0)) *
              quadrature.weight(q);
    }


    void
    assemble_penalty_matrix(unsigned int k)
    {
      AssertIndexRange(k, degree + 1);
      auto &penalty_matrix = penalty_matrices[k - 1];

      const unsigned int N               = 2 * degree + 1;
      const unsigned int n_dofs_per_cell = degree + 1;
      const unsigned int shift           = degree;

      penalty_matrix.reinit(N, N);

      std::vector<double> values_left(n_dofs_per_cell);
      std::vector<double> values_right(n_dofs_per_cell);

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        {
          values_left[i]  = polynomial_basis[k][i].value(0);
          values_right[i] = polynomial_basis[k][i].value(1);
        }



      Tensor<1, 1> normal;
      normal[0] = 1;

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
          penalty_matrix(i, j) += values_right[i] * values_right[j];

      //   fe_1d.shape_grad(numbering[i], right) *
      //   fe_1d.shape_grad(numbering[j], right);

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
          penalty_matrix(i + shift, j) -= values_left[i] * values_right[j];
      //   fe_1d.shape_grad(numbering[i], left) *
      // fe_1d.shape_grad(numbering[j], right);

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
          penalty_matrix(i, j + shift) -= values_right[i] * values_left[j];
      //   fe_1d.shape_grad(numbering[i], right) *
      //     fe_1d.shape_grad(numbering[j], left);

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
          penalty_matrix(i + shift, j + shift) +=
            values_left[i] * values_left[j];
      //   fe_1d.shape_grad(numbering[i], left) *
      //     fe_1d.shape_grad(numbering[j], left);
    }
  };

  template <int dim, typename Number>
  class TensorProductApplier
  {
  public:
    template <typename Number2>
    TensorProductApplier(const FullMatrix<Number2> &mass_matrix_,
                         const FullMatrix<Number2> &penalty_matrix_)
      : n_mass(mass_matrix_.size(0))
      , m(penalty_matrix_.size(0))
      , size(dim == 2 ? n_mass * m : n_mass * n_mass * m)
    {
      const auto copy_mat = [](const FullMatrix<Number2> &mat_in,
                               Table<2, Number> &         mat_out) -> void {
        const unsigned int nloc = mat_in.size(0);
        const unsigned int mloc = mat_in.size(1);
        mat_out.reinit(nloc, mloc);
        for (unsigned int i = 0; i < nloc; ++i)
          for (unsigned int j = 0; j < mloc; ++j)
            mat_out(i, j) = mat_in(i, j);
      };
      copy_mat(mass_matrix_, mass);
      copy_mat(penalty_matrix_, penalty);

      AssertDimension(penalty.size(0), penalty.size(1));
      AssertDimension(mass.size(0), mass.size(1));
      // FIXME:  I have no idea why you need to multiply that by 2 to get the
      // matrices to match
      for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < m; ++j)
          penalty(i, j) *= Number(2.);
    }


    void
    vmult(const ArrayView<Number> &      dst,
          const ArrayView<const Number> &src) const;

    FullMatrix<double>
    get_matrix() const;

  private:
    Table<2, Number>   mass;
    Table<2, Number>   penalty;
    const unsigned int n_mass;
    const unsigned int m;
    const unsigned int size;
  };


  template <int dim>
  void
  Generator<dim>::make_grid_and_dofs()
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
          GridTools::rotate(Tensor<1, 3, double>({0, 0, 1}), -M_PI / 2., tria);
        // FIXME!
        else if (direction == 2)
          GridTools::rotate(Tensor<1, 3, double>({0, 1, 0}), M_PI / 2., tria);
      }


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



  template <int dim>
  std::vector<unsigned int>
  Generator<dim>::get_duplicated_dofs() const
  {
    std::vector<unsigned int> duplicates;
    std::set<unsigned int>    already_in(
      cell_tensor2face_tensor_numbering[0].begin(),
      cell_tensor2face_tensor_numbering[0].end());
    for (unsigned int i = 0; i < cell_tensor2face_tensor_numbering[1].size();
         ++i)
      {
        if (std::find(already_in.begin(),
                      already_in.end(),
                      cell_tensor2face_tensor_numbering[1][i]) !=
            already_in.end())
          duplicates.push_back(i);
      }

    return duplicates;
  }


  template <int dim>
  void
  Generator<dim>::output_gnuplot()
  {
    std::map<types::global_dof_index, Point<dim>> dof_location_map;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                         dof_handler,
                                         dof_location_map);


    std::ofstream dof_location_file("dof-locations" +
                                    std::to_string(direction) + ".gnuplot");
    std::ofstream mesh_file("mesh" + std::to_string(direction) + ".gnuplot");
    GridOut().write_gnuplot(tria, mesh_file);
    DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                   dof_location_map);
  }


  template <int dim>
  void
  Generator<dim>::assemble_interface_matrix()
  {
    ghost_penalty_matrix.reinit(dof_handler.n_dofs(), dof_handler.n_dofs());


    const QGauss<dim - 1>  face_quadrature(degree + 1);
    FEInterfaceValues<dim> fe_interface_values(dof_handler.get_fe(),
                                               face_quadrature,
                                               update_gradients |
                                                 update_JxW_values |
                                                 update_normal_vectors |
                                                 update_hessians |
                                                 update_3rd_derivatives);

    // auto cell = dof_handler.begin_active();
    for (auto cell : dof_handler.active_cell_iterators())
      for (unsigned int f = 0; f < dim; ++f)
        {
          if (!cell->face(f)->at_boundary())
            {
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
                          (normal *
                           fe_interface_values.jump_in_shape_gradients(i, q) *
                           normal *
                           fe_interface_values.jump_in_shape_gradients(j, q)) *
                          fe_interface_values.JxW(q);
                      }
                }

              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

              interfece_to_tensor_product_numbering =
                local_interface_dof_indices;

              for (unsigned int i = 0; i < n_interface_dofs; ++i)
                for (unsigned int j = 0; j < n_interface_dofs; ++j)
                  ghost_penalty_matrix(local_interface_dof_indices[i],
                                       local_interface_dof_indices[j]) +=
                    local_stabilization(i, j);
            }
        }
  }


  template <int dim>
  void
  Generator<dim>::assemble_1D_mass_matrix()
  {
    const unsigned int N = degree + 1;
    mass_matrix.reinit(N, N);

    std::vector<unsigned int> numbering =
      fe_1d.get_poly_space_numbering_inverse();


    QGauss<1> quadrature(N);

    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < N; ++j)
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          mass_matrix(i, j) +=
            (fe_1d.shape_value(numbering[i], quadrature.point(q)) *
             fe_1d.shape_value(numbering[j], quadrature.point(q))) *
            quadrature.weight(q);
  }


  template <int dim>
  void
  Generator<dim>::assemble_1D_interface_matrix()
  {
    std::vector<unsigned int> numbering =
      fe_1d.get_poly_space_numbering_inverse();

    const unsigned int N               = 2 * degree + 1;
    const unsigned int n_dofs_per_cell = degree + 1;
    const unsigned int shift           = degree;
    penalty_matrix.reinit(N, N);

    Point<1> right(1);
    Point<1> left(0);

    Tensor<1, 1> normal;
    normal[0] = 1;

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        penalty_matrix(i, j) += fe_1d.shape_grad(numbering[i], right) *
                                fe_1d.shape_grad(numbering[j], right);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        penalty_matrix(i + shift, j) -= fe_1d.shape_grad(numbering[i], left) *
                                        fe_1d.shape_grad(numbering[j], right);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        penalty_matrix(i, j + shift) -= fe_1d.shape_grad(numbering[i], right) *
                                        fe_1d.shape_grad(numbering[j], left);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        penalty_matrix(i + shift, j + shift) +=
          fe_1d.shape_grad(numbering[i], left) *
          fe_1d.shape_grad(numbering[j], left);
  }


  template <int dim>
  void
  Generator<dim>::initialize()
  {
    make_grid_and_dofs();
    output_gnuplot();
    assemble_interface_matrix();
    assemble_1D_mass_matrix();
    assemble_1D_interface_matrix();
  }


  template <int dim>
  void
  Generator<dim>::print() const
  {
    ghost_penalty_matrix.print(std::cout, 10, 3);
    std::cout << std::endl;

    if (dim == 1)
      {
        auto difference = ghost_penalty_matrix;
        difference.add(-1., penalty_matrix);

        if (difference.frobenius_norm() < 1e-10)
          {
            std::cout << std::endl;
            std::cout << "Matrices match" << std::endl;
          }
        else
          std::cout << "Matrices DO NOT match" << std::endl;
      }
  }


  template <int dim, typename Number>
  void
  TensorProductApplier<dim, Number>::vmult(
    const ArrayView<Number> &      dst,
    const ArrayView<const Number> &src) const
  {
    AssertDimension(dst.size(), size);
    AssertDimension(src.size(), size);
    if constexpr (dim == 2)
      {
        AlignedVector<Number> tmp(size);

        // Compute IxA using the block structure
        for (unsigned int i = 0; i < n_mass; ++i)
          {
            for (unsigned int p = 0; p < m; ++p)
              {
                // tmp[i * m + p] = 0;
                for (unsigned int q = 0; q < m; ++q)
                  tmp[i * m + p] += penalty[p][q] * src[i * m + q];
              }
          }
        // Compute MxI using the block structure
        for (unsigned int i = 0; i < n_mass; ++i)
          {
            for (unsigned int p = 0; p < m; ++p)
              dst[i * m + p] = 0;
            for (unsigned int j = 0; j < n_mass; ++j)
              for (unsigned int p = 0; p < m; ++p)
                dst[i * m + p] += mass[i][j] * tmp[j * m + p];
          }
        return;
      }

    if constexpr (dim == 3)
      {
        AlignedVector<Number> tmp(size);
        AlignedVector<Number> tmp2(size);
        for (auto &v : dst)
          v = 0;

        // Compute IxIxA using the Kronecker product structure
        for (unsigned int i = 0; i < n_mass; ++i)
          {
            for (unsigned int j = 0; j < n_mass; ++j)
              for (unsigned int k = 0; k < m; ++k)
                //   tmp[(i * n_mass + j) * m + k] = 0;
                for (unsigned int l = 0; l < m; ++l)
                  tmp[(i * n_mass + j) * m + k] +=
                    penalty[k][l] * src[(i * n_mass + j) * m + l];
          }



        // Compute IxMxI using the Kronecker product structure
        for (unsigned int i = 0; i < n_mass; ++i)
          {
            for (unsigned int j = 0; j < n_mass; ++j)
              for (unsigned int k = 0; k < n_mass; ++k)
                for (unsigned int p = 0; p < m; ++p)
                  tmp2[(i * n_mass + j) * m + p] +=
                    mass[j][k] * tmp[(i * n_mass + k) * m + p];
          }

        // Compute MxIxI using the Kronecker product structure
        for (unsigned int i = 0; i < n_mass; ++i)
          {
            for (unsigned int j = 0; j < n_mass; ++j)
              for (unsigned int k = 0; k < n_mass; ++k)
                for (unsigned int p = 0; p < m; ++p)
                  dst[(i * n_mass + j) * m + p] +=
                    mass[i][k] * tmp2[(k * n_mass + j) * m + p];
          }
      }
  }


  template <int dim, typename Number>
  FullMatrix<double>
  TensorProductApplier<dim, Number>::get_matrix() const
  {
    static_assert(std::is_same<Number, float>::value ||
                    std::is_same<Number, double>::value,
                  "Number must be float or double");

    if constexpr (!std::is_same<Number, float>::value &&
                  !std::is_same<Number, double>::value)
      return FullMatrix<double>();

    else
      {
        FullMatrix<Number> result(size, size);
        result = 0;

        if constexpr (dim == 2)
          {
            for (unsigned int i = 0; i < n_mass; ++i)
              for (unsigned int j = 0; j < n_mass; ++j)
                for (unsigned int p = 0; p < m; ++p)
                  for (unsigned int q = 0; q < m; ++q)
                    result(i * m + p, j * m + q) = mass(i, j) * penalty(p, q);
          }
        else if constexpr (dim == 3)
          {
            for (unsigned int i1 = 0; i1 < n_mass; ++i1)
              for (unsigned int j1 = 0; j1 < n_mass; ++j1)
                for (unsigned int i2 = 0; i2 < n_mass; ++i2)
                  for (unsigned int j2 = 0; j2 < n_mass; ++j2)
                    for (unsigned int p = 0; p < m; ++p)
                      for (unsigned int q = 0; q < m; ++q)
                        result((i1 * n_mass + i2) * m + p,
                               (j1 * n_mass + j2) * m + q) =
                          mass(i1, j1) * mass(i2, j2) * penalty(p, q);
          }

        return result;
      }
  }

} // namespace GhostPenalty
#endif // __GHOST_PENALTY_OPERATOR_H__