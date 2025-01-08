#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>


namespace LaplaceOperator
{
  using namespace dealii;

  template <int dim, int degree>
  class Operator : public MatrixFreeOperators::
                     Base<dim, LinearAlgebra::distributed::Vector<double>>
  {
  public:
    using Number     = double;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

    // using VectorType                     = dealii::Vector<Number>;
    const static unsigned int n_q_points = degree + 1;

    Operator();

    void
    clear() override;

    void
    compute_diagonal() override;


  protected:
    virtual void
    apply_add(VectorType &dst, const VectorType &src) const override;


  private:
    void
    local_apply(const dealii::MatrixFree<dim, Number> &      data,
                VectorType &                                 dst,
                const VectorType &                           src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;
  };

  template <int dim, int degree>
  Operator<dim, degree>::Operator()
    : dealii::MatrixFreeOperators::Base<dim>()
  {}

  template <int dim, int degree>
  void
  Operator<dim, degree>::clear()
  {
    dealii::MatrixFreeOperators::Base<dim>::clear();
  }

  template <int dim, int degree>
  void
  Operator<dim, degree>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>());
    LinearAlgebra::distributed::Vector<Number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);


    dealii::FEEvaluation<dim, degree, n_q_points, 1, Number> fe_eval(
      *this->data);
    const unsigned int dof_per_cell = std::pow(degree + 1, dim);
    AlignedVector<VectorizedArray<Number>> local_diagonal(dof_per_cell);
    ArrayView<VectorizedArray<Number>> dof_values(fe_eval.begin_dof_values(),
                                                  dof_per_cell);
    for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
      {
        fe_eval.reinit(cell);

        for (unsigned int k = 0; k < dof_per_cell; ++k)
          { // fill fe_eval with unit vector
            for (unsigned int i = 0; i < dof_per_cell; ++i)
              dof_values[i] = 0;
            dof_values[k] = 1;

            //=======
            // A*u
            fe_eval.evaluate(EvaluationFlags::gradients);
            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
            fe_eval.integrate(EvaluationFlags::gradients);
            //=======

            // collect coputed diagonal
            local_diagonal[k] = dof_values[k];
          }
        // put diagonal back into fe_eval
        for (unsigned int i = 0; i < dof_per_cell; ++i)
          dof_values[i] = local_diagonal[i];

        fe_eval.distribute_local_to_global(inverse_diagonal);
      }

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1. / inverse_diagonal.local_element(i);
      }
  }

  template <int dim, int degree>
  void
  Operator<dim, degree>::apply_add(VectorType &dst, const VectorType &src) const
  {
    this->data->cell_loop(&Operator::local_apply, this, dst, src);
  }

  template <int dim, int degree>
  void
  Operator<dim, degree>::local_apply(
    const dealii::MatrixFree<dim, Number> &      data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    dealii::FEEvaluation<dim, degree, n_q_points, 1, Number> fe_eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
        fe_eval.integrate(EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
  }
} // namespace LaplaceOperator
#endif // LAPLACE_OPERATOR_H