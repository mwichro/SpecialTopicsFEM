#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "laplacian.h"
#include "laplacian_mf.h"

int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const unsigned int dim    = 2;
      const unsigned int degree = 1;

      Triangulation<2> triangulation;
      GridGenerator::hyper_cube(triangulation);

      LaplacianMatrixFree<dim, degree> laplace_problem(triangulation);
      laplace_problem.initialize();

      Laplacian<dim> laplace_problem2(triangulation, degree);
      laplace_problem2.intinlize();

      const auto &laplace_mf_operator = laplace_problem.get_operator();
      typename LaplacianMatrixFree<dim, degree>::VectorType dst1;
      typename LaplacianMatrixFree<dim, degree>::VectorType src;
      typename LaplacianMatrixFree<dim, degree>::VectorType dst2;
      laplace_mf_operator.initialize_dof_vector(src);
      laplace_mf_operator.initialize_dof_vector(dst1);
      laplace_mf_operator.initialize_dof_vector(dst2);

      for (unsigned int i = 0; i < src.size(); ++i)
        {
          src    = 0;
          src(i) = 1;
          laplace_mf_operator.vmult(dst1, src);
          laplace_problem2.vmult(dst2, src);
          dst1 -= dst2;
          Assert(dst1.l2_norm() < 1e-8, ExcMessage("Error in vmult"));
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}
