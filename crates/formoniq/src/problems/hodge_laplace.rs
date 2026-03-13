use crate::{
  assemble::{self, assemble_galmat, assemble_galmat_coord_aware, GalMat, GalVec},
  operators::{DofIdx, HodgeMassElmat, InnerProductWeightClosure},
};

use {
  common::linalg::petsc::{
    petsc_ghep_reduced_with_which, petsc_ghiep, petsc_ghiep_largest, petsc_saddle_point,
    GhiepReducedSolve, GhiepWhich,
  },
  ddf::{cochain::Cochain, ManifoldComplexExt},
  exterior::ExteriorGrade,
  manifold::geometry::coord::mesh::MeshCoords,
  manifold::geometry::coord::quadrature::SimplexQuadRule,
  manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex},
};

use common::linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector};
use itertools::Itertools;
use std::{collections::HashSet, mem};

use crate::operators::DofCoeff;
use manifold::topology::handle::KSimplexIdx;

pub fn solve_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_inner(
    topology,
    geometry,
    None,
    source_galvec,
    grade,
    homology_dim,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
  )
}

pub fn solve_weighted_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_inner(
    topology,
    geometry,
    None,
    source_galvec,
    grade,
    homology_dim,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
    None,
    None,
  )
}

pub fn solve_weighted_hodge_laplace_source_with_boundary_conditions(
  topology: &Complex,
  geometry: &MeshLengths,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_inner(
    topology,
    geometry,
    sigma_vec,
    u_vec,
    grade,
    homology_dim,
    Some(coords),
    qr,
    Some(weight),
    Some(k_strong_bc_predicate),
    Some(k_strong_bc_data),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_minus_one_strong_bc_data),
  )
}

/// Recommended flow when reusing assembled matrices:
/// ```text
/// let galmats = MixedGalmats::compute(&topology, &metric, grade);
/// let harmonics = solve_hodge_laplace_harmonics_with_galmats(
///   &topology, &galmats, grade, homology_dim, None, None
/// );
/// let (sigma, u, p) = solve_hodge_laplace_source_with_galmats(
///   &topology, &galmats, source, grade, homology_dim
/// );
/// ```
/// This avoids reassembling FEEC operators when solving multiple related problems.
/// Variant that reuses preassembled mixed matrices.
pub fn solve_hodge_laplace_source_with_galmats(
  topology: &Complex,
  galmats: &MixedGalmats,
  source_galvec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_with_galmats_inner(
    topology,
    galmats,
    None,
    source_galvec,
    grade,
    homology_dim,
    None,
    None,
    None,
    None,
  )
}

/// Variant that reuses preassembled mixed matrices with strong boundary conditions.
pub fn solve_hodge_laplace_source_with_galmats_and_boundary_conditions(
  topology: &Complex,
  galmats: &MixedGalmats,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_with_galmats_inner(
    topology,
    galmats,
    sigma_vec,
    u_vec,
    grade,
    homology_dim,
    Some(k_strong_bc_predicate),
    Some(k_strong_bc_data),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_minus_one_strong_bc_data),
  )
}

fn solve_hodge_laplace_source_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_minus_one_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
) -> (Cochain, Cochain, Cochain) {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr.clone(), weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

  solve_hodge_laplace_source_with_galmats_inner(
    topology,
    &galmats,
    sigma_vec,
    u_vec,
    grade,
    homology_dim,
    k_strong_bc_predicate,
    k_strong_bc_data,
    k_minus_one_strong_bc_predicate,
    k_minus_one_strong_bc_data,
  )
}

fn solve_hodge_laplace_source_with_galmats_inner(
  topology: &Complex,
  galmats: &MixedGalmats,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_minus_one_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
) -> (Cochain, Cochain, Cochain) {
  debug_assert_eq!(galmats.u_len(), topology.nsimplices(grade));
  if grade > 0 {
    debug_assert_eq!(galmats.sigma_len(), topology.nsimplices(grade - 1));
  }

  let harmonics = solve_hodge_laplace_harmonics_with_galmats(
    topology,
    galmats,
    grade,
    homology_dim,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  );

  let sigma_len = if let Some(pred) = k_minus_one_strong_bc_predicate {
    galmats.free_sigma_len(pred)
  } else {
    galmats.sigma_len()
  };

  let u_len = if let Some(pred) = k_strong_bc_predicate {
    galmats.free_u_len(pred)
  } else {
    galmats.u_len()
  };

  let sigma_vec = if let Some(sigma_vec) = sigma_vec {
    sigma_vec
  } else {
    Vector::zeros(sigma_len)
  };
  let mut owned_sigma_vec = sigma_vec.into_owned();
  let mut owned_u_vec = u_vec;

  let (system_matrix, rhs) = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
    k_minus_one_strong_bc_predicate.unwrap_or(&|_| false),
    k_minus_one_strong_bc_data.unwrap_or(&|_| 0.0),
    k_strong_bc_predicate.unwrap_or(&|_| false),
    k_strong_bc_data.unwrap_or(&|_| 0.0),
    &mut owned_sigma_vec,
    &mut owned_u_vec,
    &harmonics,
  );

  let galsol = petsc_saddle_point(&system_matrix, &rhs, harmonics.ncols() > 0).into_owned();

  let sigma = if let (Some(k_minus_one_strong_bc_predicate), Some(k_minus_one_strong_bc_data)) =
    (k_minus_one_strong_bc_predicate, k_minus_one_strong_bc_data)
  {
    let boundary_data = (0..galmats.sigma_len())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .map(|i| (i, k_minus_one_strong_bc_data(i)))
      .collect::<Vec<_>>();
    let mut temp_sigma = galsol.view_range(..sigma_len, 0).into_owned();
    assemble::reintroduce_non_homogenous_dofs_galsols(&boundary_data, &mut temp_sigma);
    Cochain::new(grade - 1, temp_sigma)
  } else {
    Cochain::new(grade - 1, galsol.view_range(..sigma_len, 0).into_owned())
  };

  let u = if let (Some(k_strong_bc_predicate), Some(k_strong_bc_data)) =
    (k_strong_bc_predicate, k_strong_bc_data)
  {
    let boundary_data = (0..galmats.u_len())
      .filter(|&i| k_strong_bc_predicate(i))
      .map(|i| (i, k_strong_bc_data(i)))
      .collect::<Vec<_>>();
    let mut temp_u = galsol
      .view_range(sigma_len..sigma_len + u_len, 0)
      .into_owned();
    assemble::reintroduce_non_homogenous_dofs_galsols(&boundary_data, &mut temp_u);
    Cochain::new(grade, temp_u)
  } else {
    Cochain::new(
      grade,
      galsol
        .view_range(sigma_len..sigma_len + u_len, 0)
        .into_owned(),
    )
  };

  let p = Cochain::new(
    grade,
    galsol.view_range(sigma_len + u_len.., 0).into_owned(),
  );
  (sigma, u, p)
}

pub fn solve_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> Matrix {
  solve_hodge_laplace_harmonics_inner(
    topology,
    geometry,
    grade,
    homology_dim,
    None,
    None,
    None,
    None,
    None,
  )
}

pub fn solve_weighted_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> Matrix {
  solve_hodge_laplace_harmonics_inner(
    topology,
    geometry,
    grade,
    homology_dim,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
  )
}

/// Variant that reuses preassembled mixed matrices.
pub fn solve_hodge_laplace_harmonics_with_galmats(
  topology: &Complex,
  galmats: &MixedGalmats,
  grade: ExteriorGrade,
  homology_dim: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> Matrix {
  debug_assert_eq!(galmats.u_len(), topology.nsimplices(grade));
  if grade > 0 {
    debug_assert_eq!(galmats.sigma_len(), topology.nsimplices(grade - 1));
  }

  if homology_dim == 0 {
    return Matrix::zeros(galmats.u_len(), 0);
  }

  let (eigenvals, _, harmonics) = solve_hodge_laplace_evp_with_galmats(
    galmats,
    homology_dim,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  );

  if !eigenvals.iter().all(|&eigenval| eigenval <= 1e-12) {
    panic!(
      "Expected zero eigenvalues for harmonic forms, but got eigenvalues: {:?}",
      eigenvals
    );
  }
  harmonics
}

fn solve_hodge_laplace_harmonics_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> Matrix {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr, weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

  solve_hodge_laplace_harmonics_with_galmats(
    topology,
    &galmats,
    grade,
    homology_dim,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  )
}

pub fn solve_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    None,
    None,
    None,
    None,
    None,
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_hodge_laplace_evp_config(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    None,
    None,
    None,
    None,
    None,
    which,
    mass_solve,
  )
}

pub fn solve_hodge_laplace_evp_largest(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    None,
    None,
    None,
    None,
    None,
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_weighted_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_weighted_hodge_laplace_evp_largest(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_weighted_hodge_laplace_evp_with_boundary_conditions(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_strong_bc_predicate),
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_weighted_hodge_laplace_evp_with_boundary_conditions_largest(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_strong_bc_predicate),
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

/// Variant that reuses preassembled mixed matrices.
pub fn solve_hodge_laplace_evp_with_galmats(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_with_galmats_impl(
    galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_hodge_laplace_evp_with_galmats_config(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_with_galmats_impl(
    galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    which,
    mass_solve,
  )
}

pub fn solve_hodge_laplace_evp_largest_with_galmats(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_with_galmats_impl(
    galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

fn solve_hodge_laplace_evp_with_galmats_impl(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  let (evals, evec_sigmas, evec_us) = if which == GhiepWhich::Smallest {
    let (lhs, sigma_len, u_len) = if let (Some(km1_pred), Some(k_pred)) =
      (k_minus_one_strong_bc_predicate, k_strong_bc_predicate)
    {
      (
        galmats.reduced_mixed_hodge_laplacian(km1_pred, k_pred),
        galmats.free_sigma_len(km1_pred),
        galmats.free_u_len(k_pred),
      )
    } else {
      (
        galmats.mixed_hodge_laplacian(),
        galmats.sigma_len(),
        galmats.u_len(),
      )
    };

    let mut rhs = CooMatrix::zeros(sigma_len + u_len, sigma_len + u_len);

    let mass_u = if let Some(k_pred) = k_strong_bc_predicate {
      let mut mass_u = galmats.mass_u().clone();
      let k_strong_bc_set = (0..mass_u.nrows())
        .filter(|&i| k_pred(i))
        .collect::<HashSet<_>>();
      assemble::drop_dofs_galmat(&k_strong_bc_set, &mut mass_u);
      mass_u
    } else {
      galmats.mass_u().clone()
    };

    for (mut r, mut c, &v) in mass_u.triplet_iter() {
      r += sigma_len;
      c += sigma_len;
      rhs.push(r, c, v);
    }

    let (eigenvals, eigenvectors) = petsc_ghiep(&(&lhs).into(), &(&rhs).into(), neigen_values);

    let eigen_sigmas = eigenvectors.rows(0, sigma_len).into_owned();
    let eigen_us = eigenvectors.rows(sigma_len, u_len).into_owned();

    (eigenvals, eigen_sigmas, eigen_us)
  } else {
    let mut mass_sigma = galmats.mass_sigma().clone();
    let mut dif_sigma = galmats.dif_sigma().clone();
    let mut codif_u = galmats.codif_u().clone();
    let mut codifdif_u = galmats.codifdif_u().clone();
    let mut mass_u = galmats.mass_u().clone();

    if let (Some(km1_pred), Some(k_pred)) = (k_minus_one_strong_bc_predicate, k_strong_bc_predicate)
    {
      // TODO migrate reduced matrix construction (for individual matrices) into galmat class
      let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
        .filter(|&i| km1_pred(i))
        .collect::<HashSet<_>>();
      let k_strongly_enforced_dofs = (0..mass_u.nrows())
        .filter(|&i| k_pred(i))
        .collect::<HashSet<_>>();

      assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
      assemble::drop_dofs_rectangular_galmat(
        &k_strongly_enforced_dofs,
        &k_minus_one_strongly_enforced_dofs,
        &mut dif_sigma,
      );
      assemble::drop_dofs_rectangular_galmat(
        &k_minus_one_strongly_enforced_dofs,
        &k_strongly_enforced_dofs,
        &mut codif_u,
      );
      assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut codifdif_u);
      assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut mass_u);
    }

    let sigma_len = mass_sigma.nrows();

    if sigma_len == 0 {
      let l = CsrMatrix::from(&codifdif_u);
      let mk = CsrMatrix::from(&mass_u);

      // TODO untested, useful only for laplace beltrami problems which are currently handled elsewhere
      let (eigenvals, eigen_us) = petsc_ghiep_largest(&l, &mk, neigen_values);
      let eigen_sigmas = Matrix::zeros(0, eigen_us.ncols());
      return (eigenvals, eigen_sigmas, eigen_us);
    }

    // Solve the generalized saddle point problem for largest eigenvalues, which
    // requires eliminating sigma to avoid infinite eigenvalues.
    let l = CsrMatrix::from(&codifdif_u);
    let d = CsrMatrix::from(&dif_sigma);
    let c = CsrMatrix::from(&codif_u);
    let mkm1 = CsrMatrix::from(&mass_sigma);
    let mk = CsrMatrix::from(&mass_u);

    let (eigenvals, eigen_sigmas, eigen_us) =
      petsc_ghep_reduced_with_which(&l, &d, &c, &mkm1, &mk, neigen_values, which, mass_solve);

    (eigenvals, eigen_sigmas, eigen_us)
  };

  (evals, evec_sigmas, evec_us)
}

fn solve_hodge_laplace_evp_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    coords,
    qr,
    weight,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    which,
    mass_solve,
  )
}

fn solve_hodge_laplace_evp_inner_with_solver(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr, weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

  solve_hodge_laplace_evp_with_galmats_impl(
    &galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    which,
    mass_solve,
  )
}

pub struct MixedGalmats {
  mass_sigma: GalMat,
  dif_sigma: GalMat,
  codif_u: GalMat,
  codifdif_u: GalMat,
  mass_u: GalMat,
}
impl MixedGalmats {
  pub fn compute(topology: &Complex, geometry: &MeshLengths, grade: ExteriorGrade) -> Self {
    let dim = topology.dim();
    assert!(grade <= dim);

    let mass_u = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, grade));
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, grade - 1));

      let exdif_sigma = topology.exterior_derivative_operator(grade - 1);
      let exdif_sigma = CsrMatrix::from(&exdif_sigma);

      let dif_sigma = &mass_u_csr * &exdif_sigma;
      let dif_sigma = CooMatrix::from(&dif_sigma);

      let codif_u = &exdif_sigma.transpose() * &mass_u_csr;
      let codif_u = CooMatrix::from(&codif_u);

      (mass_sigma, dif_sigma, codif_u)
    } else {
      (GalMat::new(0, 0), GalMat::new(0, 0), GalMat::new(0, 0))
    };

    let codifdif_u = if grade < topology.dim() {
      let mass_plus = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, grade + 1));
      let mass_plus = CsrMatrix::from(&mass_plus);
      let exdif_u = topology.exterior_derivative_operator(grade);
      let exdif_u = CsrMatrix::from(&exdif_u);
      let codifdif_u = exdif_u.transpose() * mass_plus * exdif_u;
      CooMatrix::from(&codifdif_u)
    } else {
      GalMat::new(0, 0)
    };

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    }
  }

  pub fn compute_weighted(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    coords: &MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &InnerProductWeightClosure,
  ) -> Self {
    let dim = topology.dim();
    assert!(grade <= dim);

    let mass_u = assemble_galmat_coord_aware(
      topology,
      geometry,
      HodgeMassElmat::new_weighted(dim, grade, coords, qr.clone(), weight),
    );
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = assemble_galmat_coord_aware(
        topology,
        geometry,
        HodgeMassElmat::new_weighted(dim, grade - 1, coords, qr.clone(), weight),
      );

      let exdif_sigma = topology.exterior_derivative_operator(grade - 1);
      let exdif_sigma = CsrMatrix::from(&exdif_sigma);

      let dif_sigma = &mass_u_csr * &exdif_sigma;
      let dif_sigma = CooMatrix::from(&dif_sigma);

      let codif_u = &exdif_sigma.transpose() * &mass_u_csr;
      let codif_u = CooMatrix::from(&codif_u);

      (mass_sigma, dif_sigma, codif_u)
    } else {
      (GalMat::new(0, 0), GalMat::new(0, 0), GalMat::new(0, 0))
    };

    let codifdif_u = if grade < topology.dim() {
      let mass_plus = assemble_galmat_coord_aware(
        topology,
        geometry,
        HodgeMassElmat::new_weighted(dim, grade + 1, coords, qr.clone(), weight),
      );
      let mass_plus = CsrMatrix::from(&mass_plus);
      let exdif_u = topology.exterior_derivative_operator(grade);
      let exdif_u = CsrMatrix::from(&exdif_u);
      let codifdif_u = exdif_u.transpose() * mass_plus * exdif_u;
      CooMatrix::from(&codifdif_u)
    } else {
      GalMat::new(0, 0)
    };

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    }
  }

  pub fn sigma_len(&self) -> usize {
    self.mass_sigma.nrows()
  }
  pub fn u_len(&self) -> usize {
    self.mass_u.nrows()
  }

  pub fn mass_sigma(&self) -> &GalMat {
    &self.mass_sigma
  }
  pub fn dif_sigma(&self) -> &GalMat {
    &self.dif_sigma
  }
  pub fn codif_u(&self) -> &GalMat {
    &self.codif_u
  }
  pub fn codifdif_u(&self) -> &GalMat {
    &self.codifdif_u
  }
  pub fn mass_u(&self) -> &GalMat {
    &self.mass_u
  }

  pub fn free_sigma_len(
    &self,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  ) -> usize {
    let total = self.mass_sigma.nrows();
    let n_strong_bc = (0..total)
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .count();
    total - n_strong_bc
  }

  pub fn free_u_len(&self, k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool) -> usize {
    let total = self.mass_u.nrows();
    let n_strong_bc = (0..total).filter(|&i| k_strong_bc_predicate(i)).count();
    total - n_strong_bc
  }

  pub fn mass_u_csr(&self) -> CsrMatrix {
    CsrMatrix::from(&self.mass_u)
  }

  /// Schur complement of the mixed formulation using a lumped inverse for the sigma mass matrix.
  pub fn hodge_laplacian_schur_complement_lumped(&self) -> CsrMatrix {
    if self.mass_sigma.nrows() == 0 {
      return CsrMatrix::from(&self.codifdif_u);
    }

    let mass_sigma = CsrMatrix::from(&self.mass_sigma);
    let dif_sigma = CsrMatrix::from(&self.dif_sigma);
    let codif_u = CsrMatrix::from(&self.codif_u);
    let codifdif_u = CsrMatrix::from(&self.codifdif_u);

    let mass_sigma_inv = invert_diag(&lumped_diag(&mass_sigma));
    let codif_u_scaled = scale_rows(&codif_u, &mass_sigma_inv);

    let schur = &dif_sigma * &codif_u_scaled;
    add_sparse(&codifdif_u, &schur)
  }

  pub fn mixed_hodge_laplacian(&self) -> CooMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      ..
    } = self;
    let codif_u = codif_u.clone();
    CooMatrix::block(&[&[mass_sigma, &(codif_u.neg())], &[dif_sigma, codifdif_u]])
  }

  pub fn reduced_mixed_hodge_laplacian(
    &self,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  ) -> CooMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      ..
    } = self;
    let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .collect::<HashSet<_>>();
    let k_strongly_enforced_dofs = (0..codifdif_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .collect::<HashSet<_>>();

    let mut mass_sigma = mass_sigma.clone();
    let mut dif_sigma = dif_sigma.clone();
    let mut codif_u = codif_u.clone();
    let mut codifdif_u = codifdif_u.clone();
    assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
    // TODO check if rows/cols semantics are correct
    assemble::drop_dofs_rectangular_galmat(
      &k_strongly_enforced_dofs,
      &k_minus_one_strongly_enforced_dofs,
      &mut dif_sigma,
    );
    assemble::drop_dofs_rectangular_galmat(
      &k_minus_one_strongly_enforced_dofs,
      &k_strongly_enforced_dofs,
      &mut codif_u,
    );
    assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut codifdif_u);
    CooMatrix::block(&[&[&mass_sigma, &(codif_u.neg())], &[&dif_sigma, &codifdif_u]])
  }

  pub fn mixed_hodge_laplacian_with_strong_bc_via_elimination(
    &self,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
    k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
    sigma_rhs: &mut Vector,
    u_rhs: &mut Vector,
    harmonics: &Matrix, // Assumed to already be in the reduced basis
  ) -> (CsrMatrix, Vector) {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    } = self;

    let mut mass_sigma = mass_sigma.clone();
    let mut dif_sigma = dif_sigma.clone();
    let mut codif_u_neg = codif_u.clone().neg();
    let mut codifdif_u = codifdif_u.clone();
    let mut mass_u = mass_u.clone();

    let k_minus_one_strongly_enforced_data = (0..mass_sigma.nrows())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .map(|i| (i, k_minus_one_strong_bc_data(i)))
      .collect::<Vec<_>>();

    let k_strongly_enforced_data = (0..mass_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .map(|i| (i, k_strong_bc_data(i)))
      .collect::<Vec<_>>();

    fix_dofs_coeff_strong_coo(
      &k_minus_one_strongly_enforced_data,
      &mut mass_sigma,
      sigma_rhs,
    );

    fix_dofs_coeff_strong_coo(&k_strongly_enforced_data, &mut codifdif_u, u_rhs);

    fix_dofs_coeff_strong_coo_rectangular(
      &k_minus_one_strongly_enforced_data,
      k_strong_bc_predicate,
      &mut dif_sigma,
      u_rhs,
    );

    fix_dofs_coeff_strong_coo_rectangular(
      &k_strongly_enforced_data,
      k_minus_one_strong_bc_predicate,
      &mut codif_u_neg,
      sigma_rhs,
    );

    let k_strongly_enforced_dofs = (0..mass_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .collect::<HashSet<_>>();

    let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .collect::<HashSet<_>>();

    assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
    assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut codifdif_u);
    assemble::drop_dofs_rectangular_galmat(
      &k_strongly_enforced_dofs,
      &k_minus_one_strongly_enforced_dofs,
      &mut dif_sigma,
    );
    assemble::drop_dofs_rectangular_galmat(
      &k_minus_one_strongly_enforced_dofs,
      &k_strongly_enforced_dofs,
      &mut codif_u_neg,
    );

    let k_strongly_enforced_dofs_slice =
      k_strongly_enforced_dofs.iter().cloned().collect::<Vec<_>>();
    let k_minus_one_strongly_enforced_dofs_slice = k_minus_one_strongly_enforced_dofs
      .iter()
      .cloned()
      .collect::<Vec<_>>();

    assemble::drop_dofs_galvec(&k_minus_one_strongly_enforced_dofs_slice, sigma_rhs);

    assemble::drop_dofs_galvec(&k_strongly_enforced_dofs_slice, u_rhs);

    let mut galmat = CooMatrix::block(&[&[&mass_sigma, &codif_u_neg], &[&dif_sigma, &codifdif_u]]);

    let rhs_vec = if harmonics.ncols() > 0 {
      let mut harmonics_rhs = Vector::zeros(mass_u.nrows());

      // Restricts mass_u to free dofs and makes rhs equal to -M_ID * u_D
      fix_dofs_coeff_strong_coo(&k_strongly_enforced_data, &mut mass_u, &mut harmonics_rhs);

      assemble::drop_dofs_galvec(&k_strongly_enforced_dofs_slice, &mut harmonics_rhs);

      // RHS finally equal to -H^T * M_ID * u_D
      harmonics_rhs = &harmonics.transpose() * &harmonics_rhs;

      assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut mass_u);

      let mass_u_csr = CsrMatrix::from(&mass_u);

      let mass_harmonics = &mass_u_csr * harmonics;

      galmat.grow(harmonics.ncols(), harmonics.ncols());

      let reduced_sigma_len = mass_sigma.nrows();
      let reduced_u_len = codifdif_u.nrows();

      for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols())
      {
        let v = mass_harmonics[(r, c)];
        r += reduced_sigma_len;
        c += reduced_sigma_len + reduced_u_len;
        galmat.push(r, c, v);
      }
      for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols())
      {
        let v = mass_harmonics[(r, c)];
        // transpose
        mem::swap(&mut r, &mut c);
        r += reduced_sigma_len + reduced_u_len;
        c += reduced_sigma_len;
        galmat.push(r, c, v);
      }
      na::stack![
        sigma_rhs;
        u_rhs;
        harmonics_rhs;
      ]
    } else {
      na::stack![
        sigma_rhs;
        u_rhs;
      ]
    };

    let system_matrix = CsrMatrix::from(&galmat);

    (system_matrix, rhs_vec)
  }
}

fn lumped_diag(mat: &CsrMatrix) -> Vec<f64> {
  let mut diag = vec![0.0; mat.nrows()];
  for (row, _col, value) in mat.triplet_iter() {
    diag[row] += *value;
  }
  diag
}

fn invert_diag(diag: &[f64]) -> Vec<f64> {
  let eps = 1e-12;
  diag
    .iter()
    .map(|v| if v.abs() < eps { 0.0 } else { 1.0 / v })
    .collect()
}

fn scale_rows(mat: &CsrMatrix, row_scales: &[f64]) -> CsrMatrix {
  assert_eq!(mat.nrows(), row_scales.len());
  let mut coo = CooMatrix::new(mat.nrows(), mat.ncols());
  for (row, col, value) in mat.triplet_iter() {
    let scaled = *value * row_scales[row];
    if scaled != 0.0 {
      coo.push(row, col, scaled);
    }
  }
  CsrMatrix::from(&coo)
}

fn add_sparse(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
  assert_eq!(a.nrows(), b.nrows());
  assert_eq!(a.ncols(), b.ncols());

  let mut coo = CooMatrix::from(a);
  for (row, col, value) in b.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}
// TODO same functionality as assemble::fix_dofs_coeff but different implementation
// Benchmark then decide which one to keep
pub fn fix_dofs_coeff_strong_coo(
  dof_coeffs: &[(DofIdx, f64)],
  galmat: &mut GalMat,
  galvec: &mut Vector,
) {
  let ndofs = galmat.nrows();

  let mut fixed_val: Vec<Option<f64>> = vec![None; ndofs];
  for &(i, v) in dof_coeffs {
    fixed_val[i] = Some(v);
  }

  let mut new_mat = GalMat::new(ndofs, ndofs);

  for (r, c, a_ref) in galmat.triplet_iter() {
    let a = *a_ref;

    if fixed_val[r].is_some() {
      continue;
    }

    if let Some(vc) = fixed_val[c] {
      // Move known contribution to RHS: b_r -= A_{r,c} * v_c
      galvec[r] -= a * vc;
      continue;
    }

    // Free/free coupling survives.
    new_mat.push(r, c, a);
  }

  // Impose Dirichlet rows: set b_i = v_i and set A_{i,i} = 1.
  for &(i, v) in dof_coeffs {
    galvec[i] = v;
    new_mat.push(i, i, 1.0);
  }

  *galmat = new_mat;
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;
  use manifold::gen::cartesian::CartesianMeshInfo;

  #[test]
  fn schur_complement_lumped_dimensions() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let mass_u = galmats.mass_u_csr();
    let laplacian = galmats.hodge_laplacian_schur_complement_lumped();

    assert!(mass_u.nrows() > 0);
    assert_eq!(mass_u.nrows(), mass_u.ncols());
    assert_eq!(laplacian.nrows(), mass_u.nrows());
    assert_eq!(laplacian.ncols(), mass_u.ncols());
  }

  #[test]
  fn harmonics_with_galmats_zero_dim() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let harmonics =
      solve_hodge_laplace_harmonics_with_galmats(&topology, &galmats, 1, 0, None, None);

    assert_eq!(harmonics.ncols(), 0);
    assert_eq!(harmonics.nrows(), galmats.u_len());
  }
}

pub fn fix_dofs_coeff_strong_coo_rectangular(
  col_dof_coeffs: &[(DofIdx, f64)],
  row_predicate: &dyn Fn(DofIdx) -> bool,
  galmat: &mut GalMat,
  galvec: &mut Vector,
) {
  let nrows = galmat.nrows();
  let ncols = galmat.ncols();

  let mut fixed_val: Vec<Option<f64>> = vec![None; ncols];
  for &(i, v) in col_dof_coeffs {
    fixed_val[i] = Some(v);
  }

  let mut excluded_rows: Vec<bool> = vec![false; nrows];
  for i in 0..nrows {
    if row_predicate(i) {
      excluded_rows[i] = true;
    }
  }

  let mut new_mat = GalMat::new(nrows, ncols);

  for (r, c, a_ref) in galmat.triplet_iter() {
    let a = *a_ref;

    if excluded_rows[r] {
      continue;
    }

    if let Some(vc) = fixed_val[c] {
      // Move known contribution to RHS: b_r -= A_{r,c} * v_c
      galvec[r] -= a * vc;
      continue;
    }

    // Free/free coupling survives.
    new_mat.push(r, c, a);
  }

  // Impose Dirichlet rows: set b_i = v_i and set A_{i,i} = 1.
  // for &(i, v) in col_dof_coeffs {
  //   galvec[i] = v;
  //   new_mat.push(i, i, 1.0);
  // }

  *galmat = new_mat;
}
