#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

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
