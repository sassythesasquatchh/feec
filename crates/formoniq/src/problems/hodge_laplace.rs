use crate::{
  assemble::{self, assemble_galmat, assemble_galmat_coord_aware, GalMat, GalVec},
  operators::{DofIdx, HodgeMassElmat, InnerProductWeightClosure},
};

use {
  common::linalg::petsc::{petsc_ghiep, petsc_saddle_point},
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
  let harmonics = solve_hodge_laplace_harmonics_inner(
    topology,
    geometry,
    grade,
    homology_dim,
    coords,
    qr.clone(),
    weight,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  );

  println!("Harmonics computed.");

  // TODO subtract harmonic projection from the rhs galvec if homology_dim > 0

  // TODO The galmats are already built when computing the harmonics, so rebuilding them here is inefficient
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr.clone(), weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

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
    Vector::zeros(galmats.sigma_len())
  };
  let mut owned_sigma_vec = sigma_vec.into_owned();
  let mut owned_u_vec = u_vec.into_owned();

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
  if homology_dim == 0 {
    let nwhitneys = topology.nsimplices(grade);
    return Matrix::zeros(nwhitneys, 0);
  }

  let (eigenvals, _, harmonics) = solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    homology_dim,
    coords,
    qr,
    weight,
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
  )
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
) -> (Vector, Matrix, Matrix) {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr, weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

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
  let mass_u = if let Some(k_strong_bc_predicate) = k_strong_bc_predicate {
    let mut mass_u = galmats.mass_u.clone();
    let k_strong_bc_set = (0..mass_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .collect::<HashSet<_>>();
    assemble::drop_dofs_galmat(&k_strong_bc_set, &mut mass_u);
    mass_u
  } else {
    galmats.mass_u
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
