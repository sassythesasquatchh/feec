use common::linalg::{
  faer::{FaerCholesky, FaerLu},
  nalgebra::{CooMatrix, CsrMatrix, Vector},
};

#[derive(Debug, Clone, Copy)]
pub struct NewtonOptions {
  pub max_iters: usize,
  pub decrement_tol: f64,
  pub line_search_max_iters: usize,
  pub line_search_reduction: f64,
  pub sufficient_decrease: f64,
}

impl NewtonOptions {
  pub fn standard(max_iters: usize, decrement_tol: f64) -> Self {
    Self {
      max_iters,
      decrement_tol,
      line_search_max_iters: 10,
      line_search_reduction: 0.5,
      sufficient_decrease: 1e-4,
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub enum NonlinearSolveStatus {
  Converged { iterations: usize, decrement: f64 },
  MaxIters { iterations: usize, decrement: f64 },
}

pub fn newton_solve<F>(
  mut u: Vector,
  mut residual_and_jacobian: F,
  opts: NewtonOptions,
) -> (Vector, NonlinearSolveStatus)
where
  F: FnMut(&Vector) -> (Vector, CooMatrix),
{
  let mut status = NonlinearSolveStatus::MaxIters {
    iterations: 0,
    decrement: f64::INFINITY,
  };

  for iter in 0..opts.max_iters {
    let (residual, jacobian) = residual_and_jacobian(&u);
    let decrement = residual.norm();

    if decrement < opts.decrement_tol {
      status = NonlinearSolveStatus::Converged {
        iterations: iter,
        decrement,
      };
      return (u, status);
    }

    let jacobian = CsrMatrix::from(&jacobian);
    let solver = FaerLu::new(jacobian);
    let step = solver.solve(&(-&residual));

    let mut alpha = 1.0;
    let mut trial = u.clone();
    let mut trial_residual_norm = decrement;

    for _ in 0..opts.line_search_max_iters {
      trial = &u + alpha * &step;
      let (trial_residual, _) = residual_and_jacobian(&trial);
      trial_residual_norm = trial_residual.norm();
      if trial_residual_norm <= (1.0 - opts.sufficient_decrease * alpha) * decrement {
        break;
      }
      alpha *= opts.line_search_reduction;
    }

    u = trial;

    status = NonlinearSolveStatus::MaxIters {
      iterations: iter + 1,
      decrement: trial_residual_norm,
    };
  }

  (u, status)
}

pub fn gauss_newton_solve<F>(
  mut u: Vector,
  mut residual_and_jacobian: F,
  opts: NewtonOptions,
) -> (Vector, NonlinearSolveStatus)
where
  F: FnMut(&Vector) -> (Vector, CooMatrix),
{
  let mut status = NonlinearSolveStatus::MaxIters {
    iterations: 0,
    decrement: f64::INFINITY,
  };

  for iter in 0..opts.max_iters {
    let (residual, jacobian) = residual_and_jacobian(&u);
    let decrement = residual.norm();
    if decrement < opts.decrement_tol {
      status = NonlinearSolveStatus::Converged {
        iterations: iter,
        decrement,
      };
      return (u, status);
    }

    let jacobian = CsrMatrix::from(&jacobian);
    let jt = jacobian.transpose();
    let normal_matrix = &jt * &jacobian;
    let rhs = &jt * &(-&residual);
    let solver = FaerCholesky::new(normal_matrix);
    let step = solver.solve(&rhs);

    let mut alpha = 1.0;
    let mut trial = u.clone();
    let mut trial_residual_norm = decrement;

    for _ in 0..opts.line_search_max_iters {
      trial = &u + alpha * &step;
      let (trial_residual, _) = residual_and_jacobian(&trial);
      trial_residual_norm = trial_residual.norm();
      if trial_residual_norm <= (1.0 - opts.sufficient_decrease * alpha) * decrement {
        break;
      }
      alpha *= opts.line_search_reduction;
    }

    u = trial;
    status = NonlinearSolveStatus::MaxIters {
      iterations: iter + 1,
      decrement: trial_residual_norm,
    };
  }

  (u, status)
}
