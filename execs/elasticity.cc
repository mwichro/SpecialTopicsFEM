// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the bare-dealii-app application, based on the
// deal.II library.
//
// The bare-dealii-app application is free software; you can use it,
// redistribute it, and/or modify it under the terms of the Apache-2.0 License
// WITH LLVM-exception as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md
// at the top level of the bare-dealii-app distribution.
//
// ---------------------------------------------------------------------

#include "elasticity.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

using namespace dealii;

int
main()
{
  try
    {
      Triangulation<DEAL_DIMENSION> triangulation;
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(3);
      Elasticity<DEAL_DIMENSION> elastic_problem(triangulation, 2);
      elastic_problem.intinlize();
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
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
