#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>


namespace LaplaceOperator
{
  using namespace dealii;

  template <int dim, int degree>
  class Operator : public MatrixFreeOperators::Base<dim>
  {
  public:
    using Number     = double;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
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
    AssertThrow(false, ExcMessage("Not implemented"));
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