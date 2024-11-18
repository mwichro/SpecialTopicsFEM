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

#ifndef dealii_laplacian_h
#define dealii_laplacian_h

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <fstream>

#include "laplace_operator.h"

using namespace dealii;

template <int dim, int degree>
class LaplacianMatrixFree
{
public:
  using Number     = double;
  using VectorType = Vector<Number>;

  using OperatorType = LaplaceOperator::Operator<dim, degree>;

  LaplacianMatrixFree(const Triangulation<dim> &tria);

  const auto &
  get_operator() const
  {
    return system_matrix;
  }

  void
  initialize();

private:
  void
  setup_system();
  void
  solve();

  void
  output_results(const unsigned int cycle) const;

  const Triangulation<dim> &tria;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;


  AffineConstraints<double> constraints;


  std::shared_ptr<MatrixFree<dim, Number>> matrix_free_storage;
  OperatorType                             system_matrix;



  VectorType solution;
  VectorType system_rhs;
};

template <int dim, int degree>
LaplacianMatrixFree<dim, degree>::LaplacianMatrixFree(
  const Triangulation<dim> &tria)
  : tria(tria)
  , fe(degree)
  , dof_handler(tria)
{}

template <int dim, int degree>
void
LaplacianMatrixFree<dim, degree>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  constraints.reinit();
  constraints.close();

  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::none;
  additional_data.mapping_update_flags = update_gradients | update_JxW_values;

  matrix_free_storage = std::make_shared<MatrixFree<dim, Number>>();

  MappingQ1<dim> mapping;
  matrix_free_storage->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(fe.degree + 1),
                              additional_data);

  matrix_free_storage->initialize_dof_vector(solution);
  matrix_free_storage->initialize_dof_vector(system_rhs);

  system_matrix.initialize(matrix_free_storage);
}


template <int dim, int degree>
void
LaplacianMatrixFree<dim, degree>::initialize()
{
  setup_system();
}
#endif