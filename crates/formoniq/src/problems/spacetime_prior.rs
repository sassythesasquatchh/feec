use crate::assemble::{
  assemble_whitney_2form_projected_sparse_inverse_galmat,
  assemble_whitney_projected_sparse_inverse_galmat,
};
use crate::problems::hodge_laplace::MixedGalmats;
use crate::problems::laplace_beltrami::LaplaceBeltramiGalmats;
use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Matrix, Vector};
use feg_core::{
  BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, FixedDof, SoftBoundaryConstraint,
  SparseTriplet, SparseTripletMatrix, SpatialPriorSlice, StateLayout,
};
use manifold::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, handle::KSimplexIdx},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy)]
pub struct ScalarPriorConfig {
  pub kappa: f64,
  pub tau: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hodge1MassInverse {
  RowSumLumped,
  Nc1ProjectedSparseInverse,
}

#[derive(Debug, Clone, Copy)]
pub struct Hodge1PriorConfig {
  pub kappa: f64,
  pub tau: f64,
  pub mass_inverse: Hodge1MassInverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hodge2MassInverse {
  ExactTopDegreeDiagonalOrProjectedNc2,
}

#[derive(Debug, Clone, Copy)]
pub struct Hodge2PriorConfig {
  pub kappa: f64,
  pub tau: f64,
  pub mass_inverse: Hodge2MassInverse,
}

pub fn build_spatial_prior_slice_0form(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &BoundarySpec,
  config: ScalarPriorConfig,
) -> Result<SpatialPriorSlice, String> {
  ensure_no_soft_auxiliary(&boundary.auxiliary_regions)?;

  let galmats = LaplaceBeltramiGalmats::compute(topology, geometry);
  let full_mass = galmats.mass_csr();
  let full_laplacian = galmats.stiffness_csr();
  let layout = build_state_layout(full_mass.nrows(), &boundary.state_regions)?;
  let mass = reduce_square_with_layout(&full_mass, &layout)?;
  let laplacian = reduce_square_with_layout(&full_laplacian, &layout)?;
  let drift = add_sparse(&laplacian, &scale_matrix(&mass, config.kappa * config.kappa));
  let precision = build_row_sum_precision(&mass, &drift, config.tau);
  let soft_boundary_constraints = build_soft_constraints(&boundary.state_regions, &layout)?;

  Ok(SpatialPriorSlice {
    mass: csr_to_triplet(&mass),
    drift: csr_to_triplet(&drift),
    initial_precision: csr_to_triplet(&precision),
    driving_noise_precision: csr_to_triplet(&precision),
    layout,
    soft_boundary_constraints,
  })
}

pub fn build_spatial_prior_slice_1form(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &BoundarySpec,
  config: Hodge1PriorConfig,
) -> Result<SpatialPriorSlice, String> {
  let galmats = MixedGalmats::compute(topology, geometry, 1);
  let (mass, laplacian, layout) = build_reduced_hodge_slice(
    &galmats,
    boundary,
    Some(config.mass_inverse),
    None,
    topology,
    geometry,
  )?;
  let drift = add_sparse(&laplacian, &scale_matrix(&mass, config.kappa * config.kappa));
  let mass_inverse = match config.mass_inverse {
    Hodge1MassInverse::RowSumLumped => diag_matrix(&invert_diag(&lumped_diag(&mass))),
    Hodge1MassInverse::Nc1ProjectedSparseInverse => {
      let projected = CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
        topology, geometry,
      ));
      reduce_square_with_layout(&projected, &layout)?
    }
  };
  let precision = build_precision_from_mass_inverse(&drift, &mass_inverse, config.tau);
  let soft_boundary_constraints = build_soft_constraints(&boundary.state_regions, &layout)?;

  Ok(SpatialPriorSlice {
    mass: csr_to_triplet(&mass),
    drift: csr_to_triplet(&drift),
    initial_precision: csr_to_triplet(&precision),
    driving_noise_precision: csr_to_triplet(&precision),
    layout,
    soft_boundary_constraints,
  })
}

pub fn build_spatial_prior_slice_2form(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &BoundarySpec,
  config: Hodge2PriorConfig,
) -> Result<SpatialPriorSlice, String> {
  let galmats = MixedGalmats::compute(topology, geometry, 2);
  let (mass, laplacian, layout) = build_reduced_hodge_slice(
    &galmats,
    boundary,
    None,
    Some(config.mass_inverse),
    topology,
    geometry,
  )?;
  let drift = add_sparse(&laplacian, &scale_matrix(&mass, config.kappa * config.kappa));
  let mass_inverse = match config.mass_inverse {
    Hodge2MassInverse::ExactTopDegreeDiagonalOrProjectedNc2 => match topology.dim() {
      2 => diag_matrix(&invert_diag(&matrix_diag(&mass))),
      3 => {
        let projected = CsrMatrix::from(&assemble_whitney_2form_projected_sparse_inverse_galmat(
          topology, geometry,
        ));
        reduce_square_with_layout(&projected, &layout)?
      }
      dim => {
        return Err(format!(
          "2-form mass inverse is only implemented for intrinsic mesh dimensions 2 and 3, got {dim}"
        ))
      }
    },
  };
  let precision = build_precision_from_mass_inverse(&drift, &mass_inverse, config.tau);
  let soft_boundary_constraints = build_soft_constraints(&boundary.state_regions, &layout)?;

  Ok(SpatialPriorSlice {
    mass: csr_to_triplet(&mass),
    drift: csr_to_triplet(&drift),
    initial_precision: csr_to_triplet(&precision),
    driving_noise_precision: csr_to_triplet(&precision),
    layout,
    soft_boundary_constraints,
  })
}

fn build_reduced_hodge_slice(
  galmats: &MixedGalmats,
  boundary: &BoundarySpec,
  mass_inverse_1form: Option<Hodge1MassInverse>,
  mass_inverse_2form: Option<Hodge2MassInverse>,
  topology: &Complex,
  geometry: &MeshLengths,
) -> Result<(CsrMatrix, CsrMatrix, StateLayout), String> {
  ensure_no_soft_auxiliary(&boundary.auxiliary_regions)?;

  let layout = build_state_layout(galmats.u_len(), &boundary.state_regions)?;
  let auxiliary_fixed = collect_fixed_dofs(galmats.sigma_len(), &boundary.auxiliary_regions)?;
  let state_fixed = layout.fixed_dofs.clone();

  let auxiliary_fixed_map = fixed_map(galmats.sigma_len(), &auxiliary_fixed)?;
  let state_fixed_map = fixed_map(galmats.u_len(), &state_fixed)?;

  let auxiliary_fixed_set = auxiliary_fixed
    .iter()
    .map(|entry| entry.index)
    .collect::<BTreeSet<_>>();
  let state_fixed_set = state_fixed
    .iter()
    .map(|entry| entry.index)
    .collect::<BTreeSet<_>>();

  let reduced_u_len = layout.reduced_dimension();
  let harmonics = Matrix::zeros(reduced_u_len, 0);
  let sigma_predicate = |kidx: KSimplexIdx| auxiliary_fixed_set.contains(&kidx);
  let state_predicate = |kidx: KSimplexIdx| state_fixed_set.contains(&kidx);
  let sigma_data = |kidx: KSimplexIdx| auxiliary_fixed_map[kidx].unwrap_or(0.0);
  let state_data = |kidx: KSimplexIdx| state_fixed_map[kidx].unwrap_or(0.0);

  let mut sigma_rhs = Vector::zeros(galmats.sigma_len());
  let mut u_rhs = Vector::zeros(galmats.u_len());
  let (reduced_mixed, _reduced_rhs) = galmats
    .mixed_hodge_laplacian_with_strong_bc_via_elimination(
      &sigma_predicate,
      &sigma_data,
      &state_predicate,
      &state_data,
      &mut sigma_rhs,
      &mut u_rhs,
      &harmonics,
    );

  let reduced_sigma_len = galmats.free_sigma_len(&sigma_predicate);
  let (mass_sigma, a12, a21, codifdif_u) =
    split_reduced_mixed_blocks(&reduced_mixed, reduced_sigma_len, reduced_u_len);
  let mass_u = reduce_square_with_layout(&CsrMatrix::from(galmats.mass_u()), &layout)?;

  let sigma_inverse = if mass_sigma.nrows() == 0 {
    CsrMatrix::from(&CooMatrix::new(0, 0))
  } else if let Some(strategy) = mass_inverse_1form {
    match strategy {
      Hodge1MassInverse::RowSumLumped => diag_matrix(&invert_diag(&lumped_diag(&mass_sigma))),
      Hodge1MassInverse::Nc1ProjectedSparseInverse => {
        let projected = CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
          topology, geometry,
        ));
        let sigma_layout = layout_from_fixed(galmats.sigma_len(), &auxiliary_fixed)?;
        reduce_square_with_layout(&projected, &sigma_layout)?
      }
    }
  } else if let Some(strategy) = mass_inverse_2form {
    match strategy {
      Hodge2MassInverse::ExactTopDegreeDiagonalOrProjectedNc2 => match topology.dim() {
        2 => diag_matrix(&invert_diag(&matrix_diag(&mass_sigma))),
        3 => {
          let projected = CsrMatrix::from(&assemble_whitney_2form_projected_sparse_inverse_galmat(
            topology, geometry,
          ));
          let sigma_layout = layout_from_fixed(galmats.sigma_len(), &auxiliary_fixed)?;
          reduce_square_with_layout(&projected, &sigma_layout)?
        }
        dim => {
          return Err(format!(
            "2-form mixed slice is only implemented for intrinsic mesh dimensions 2 and 3, got {dim}"
          ))
        }
      },
    }
  } else {
    return Err("missing Hodge mass inverse strategy".to_string());
  };

  let laplacian = if mass_sigma.nrows() == 0 {
    codifdif_u
  } else {
    let schur_mid = &a21 * &sigma_inverse;
    let schur = schur_mid * &scale_matrix(&a12, -1.0);
    add_sparse(&codifdif_u, &schur)
  };

  Ok((mass_u, laplacian, layout))
}

fn ensure_no_soft_auxiliary(regions: &[BoundaryRegionSpec]) -> Result<(), String> {
  if let Some(region) = regions
    .iter()
    .find(|region| matches!(region.treatment, BoundaryTreatment::SoftEssential { .. }))
  {
    return Err(format!(
      "soft auxiliary boundary conditions are not supported for reduced k-form spacetime priors; region '{}' must be hard or natural",
      region.name
    ));
  }
  Ok(())
}

fn build_state_layout(
  full_dimension: usize,
  state_regions: &[BoundaryRegionSpec],
) -> Result<StateLayout, String> {
  let fixed_dofs = collect_fixed_dofs(full_dimension, state_regions)?;
  layout_from_fixed(full_dimension, &fixed_dofs)
}

fn layout_from_fixed(full_dimension: usize, fixed_dofs: &[FixedDof]) -> Result<StateLayout, String> {
  let fixed_indices = fixed_dofs.iter().map(|entry| entry.index).collect::<BTreeSet<_>>();
  let active_dofs = (0..full_dimension)
    .filter(|index| !fixed_indices.contains(index))
    .collect::<Vec<_>>();
  Ok(StateLayout::new(
    full_dimension,
    active_dofs,
    fixed_dofs.to_vec(),
  ))
}

fn fixed_map(full_dimension: usize, fixed_dofs: &[FixedDof]) -> Result<Vec<Option<f64>>, String> {
  let mut fixed_map = vec![None; full_dimension];
  for fixed in fixed_dofs {
    if fixed_map[fixed.index].is_some() {
      return Err(format!("duplicate fixed dof {}", fixed.index));
    }
    fixed_map[fixed.index] = Some(fixed.value);
  }
  Ok(fixed_map)
}

fn collect_fixed_dofs(
  full_dimension: usize,
  regions: &[BoundaryRegionSpec],
) -> Result<Vec<FixedDof>, String> {
  let mut fixed = BTreeMap::<usize, f64>::new();
  for region in regions {
    if !matches!(region.treatment, BoundaryTreatment::HardEssential) {
      continue;
    }
    validate_region(region, full_dimension)?;
    for (&dof, &value) in region.dofs.iter().zip(region.values.iter()) {
      if fixed.insert(dof, value).is_some() {
        return Err(format!("dof {dof} is fixed by multiple hard-essential regions"));
      }
    }
  }
  Ok(fixed
    .into_iter()
    .map(|(index, value)| FixedDof { index, value })
    .collect())
}

fn build_soft_constraints(
  regions: &[BoundaryRegionSpec],
  layout: &StateLayout,
) -> Result<Vec<SoftBoundaryConstraint>, String> {
  let reduced_map = reduced_index_map(layout);
  let fixed_set = layout
    .fixed_dofs
    .iter()
    .map(|entry| entry.index)
    .collect::<BTreeSet<_>>();
  let mut constraints = Vec::new();

  for region in regions {
    let BoundaryTreatment::SoftEssential { variance } = region.treatment else {
      continue;
    };
    validate_region(region, layout.full_dimension)?;

    let mut operator = SparseTripletMatrix::new(region.dofs.len(), layout.reduced_dimension());
    for (row, dof) in region.dofs.iter().copied().enumerate() {
      if fixed_set.contains(&dof) {
        return Err(format!(
          "soft-essential region '{}' references hard-fixed dof {dof}",
          region.name
        ));
      }
      let Some(reduced_index) = reduced_map[dof] else {
        return Err(format!(
          "dof {dof} from soft-essential region '{}' is not active in the reduced layout",
          region.name
        ));
      };
      operator.push(row, reduced_index, 1.0);
    }
    constraints.push(SoftBoundaryConstraint {
      operator,
      target: region.values.clone(),
      variance,
    });
  }

  Ok(constraints)
}

fn reduced_index_map(layout: &StateLayout) -> Vec<Option<usize>> {
  let mut map = vec![None; layout.full_dimension];
  for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
    map[full] = Some(reduced);
  }
  map
}

fn validate_region(region: &BoundaryRegionSpec, full_dimension: usize) -> Result<(), String> {
  if region.dofs.len() != region.values.len() {
    return Err(format!(
      "boundary region '{}' has mismatched dof/value lengths",
      region.name
    ));
  }
  if let BoundaryTreatment::SoftEssential { variance } = region.treatment {
    if !variance.is_finite() || variance <= 0.0 {
      return Err(format!(
        "soft-essential region '{}' must have positive finite variance",
        region.name
      ));
    }
  }
  for dof in &region.dofs {
    if *dof >= full_dimension {
      return Err(format!(
        "boundary region '{}' references dof {} outside dimension {}",
        region.name,
        dof,
        full_dimension
      ));
    }
  }
  Ok(())
}

fn split_reduced_mixed_blocks(
  matrix: &CsrMatrix,
  reduced_sigma_len: usize,
  reduced_u_len: usize,
) -> (CsrMatrix, CsrMatrix, CsrMatrix, CsrMatrix) {
  let mut mass_sigma = CooMatrix::new(reduced_sigma_len, reduced_sigma_len);
  let mut a12 = CooMatrix::new(reduced_sigma_len, reduced_u_len);
  let mut a21 = CooMatrix::new(reduced_u_len, reduced_sigma_len);
  let mut k_matrix = CooMatrix::new(reduced_u_len, reduced_u_len);
  let u_offset = reduced_sigma_len;
  let total = reduced_sigma_len + reduced_u_len;

  for (row, col, value) in matrix.triplet_iter() {
    if row >= total || col >= total {
      continue;
    }
    if row < reduced_sigma_len {
      if col < reduced_sigma_len {
        mass_sigma.push(row, col, *value);
      } else {
        a12.push(row, col - u_offset, *value);
      }
    } else if col < reduced_sigma_len {
      a21.push(row - u_offset, col, *value);
    } else {
      k_matrix.push(row - u_offset, col - u_offset, *value);
    }
  }

  (
    CsrMatrix::from(&mass_sigma),
    CsrMatrix::from(&a12),
    CsrMatrix::from(&a21),
    CsrMatrix::from(&k_matrix),
  )
}

fn reduce_square_with_layout(matrix: &CsrMatrix, layout: &StateLayout) -> Result<CsrMatrix, String> {
  if matrix.nrows() != layout.full_dimension || matrix.ncols() != layout.full_dimension {
    return Err(format!(
      "matrix reduction expected a {}x{} operator, got {}x{}",
      layout.full_dimension,
      layout.full_dimension,
      matrix.nrows(),
      matrix.ncols()
    ));
  }
  let reduced_map = reduced_index_map(layout);
  let mut reduced = CooMatrix::new(layout.reduced_dimension(), layout.reduced_dimension());
  for (row, col, value) in matrix.triplet_iter() {
    let Some(reduced_row) = reduced_map[row] else {
      continue;
    };
    let Some(reduced_col) = reduced_map[col] else {
      continue;
    };
    reduced.push(reduced_row, reduced_col, *value);
  }
  Ok(CsrMatrix::from(&reduced))
}

fn build_row_sum_precision(mass: &CsrMatrix, drift: &CsrMatrix, tau: f64) -> CsrMatrix {
  let mass_inverse = diag_matrix(&invert_diag(&lumped_diag(mass)));
  build_precision_from_mass_inverse(drift, &mass_inverse, tau)
}

fn build_precision_from_mass_inverse(
  drift: &CsrMatrix,
  mass_inverse: &CsrMatrix,
  tau: f64,
) -> CsrMatrix {
  let middle = mass_inverse * drift;
  let precision = drift * &middle;
  if (tau - 1.0).abs() <= f64::EPSILON {
    precision
  } else {
    scale_matrix(&precision, tau * tau)
  }
}

fn lumped_diag(mat: &CsrMatrix) -> Vec<f64> {
  let mut diag = vec![0.0; mat.nrows()];
  for (row, _col, value) in mat.triplet_iter() {
    diag[row] += *value;
  }
  diag
}

fn matrix_diag(mat: &CsrMatrix) -> Vec<f64> {
  let mut diag = vec![0.0; mat.nrows()];
  for (row, col, value) in mat.triplet_iter() {
    if row == col {
      diag[row] += *value;
    }
  }
  diag
}

fn invert_diag(diag: &[f64]) -> Vec<f64> {
  let eps = 1e-12;
  diag
    .iter()
    .map(|value| if value.abs() < eps { 0.0 } else { 1.0 / value })
    .collect()
}

fn diag_matrix(diag: &[f64]) -> CsrMatrix {
  let mut coo = CooMatrix::new(diag.len(), diag.len());
  for (index, value) in diag.iter().copied().enumerate() {
    if value != 0.0 {
      coo.push(index, index, value);
    }
  }
  CsrMatrix::from(&coo)
}

fn scale_matrix(matrix: &CsrMatrix, scale: f64) -> CsrMatrix {
  let mut coo = CooMatrix::new(matrix.nrows(), matrix.ncols());
  for (row, col, value) in matrix.triplet_iter() {
    let scaled = *value * scale;
    if scaled != 0.0 {
      coo.push(row, col, scaled);
    }
  }
  CsrMatrix::from(&coo)
}

fn add_sparse(lhs: &CsrMatrix, rhs: &CsrMatrix) -> CsrMatrix {
  assert_eq!(lhs.nrows(), rhs.nrows());
  assert_eq!(lhs.ncols(), rhs.ncols());
  let mut coo = CooMatrix::from(lhs);
  for (row, col, value) in rhs.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}

fn csr_to_triplet(matrix: &CsrMatrix) -> SparseTripletMatrix {
  SparseTripletMatrix::from_triplets(
    matrix.nrows(),
    matrix.ncols(),
    matrix.triplet_iter().map(|(row, col, value)| SparseTriplet {
      row,
      col,
      value: *value,
    }),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use manifold::gen::cartesian::CartesianMeshInfo;

  fn mixed_bc_1form_boundary() -> (Complex, MeshLengths, BoundarySpec) {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);

    let boundary_edges = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .take(2)
      .map(|simp| simp.kidx)
      .collect::<Vec<_>>();
    let boundary_vertices = topology
      .boundary_subcomplex_simplices(0)
      .into_iter()
      .take(2)
      .map(|simp| simp.kidx)
      .collect::<Vec<_>>();

    let boundary = BoundarySpec::default()
      .with_state_region(BoundaryRegionSpec::new(
        "hard-u",
        boundary_edges.clone(),
        vec![0.0; boundary_edges.len()],
        BoundaryTreatment::HardEssential,
      ))
      .with_auxiliary_region(BoundaryRegionSpec::new(
        "hard-sigma",
        boundary_vertices,
        vec![0.0; 2],
        BoundaryTreatment::HardEssential,
      ));

    (topology, geometry, boundary)
  }

  #[test]
  fn hard_1form_slice_matches_manual_reduced_schur_operator() {
    let (topology, geometry, boundary) = mixed_bc_1form_boundary();
    let slice = build_spatial_prior_slice_1form(
      &topology,
      &geometry,
      &boundary,
      Hodge1PriorConfig {
        kappa: 0.0,
        tau: 1.0,
        mass_inverse: Hodge1MassInverse::RowSumLumped,
      },
    )
    .expect("1-form slice should assemble");

    let galmats = MixedGalmats::compute(&topology, &geometry, 1);
    let hard_u = boundary.state_regions[0]
      .dofs
      .iter()
      .copied()
      .collect::<BTreeSet<_>>();
    let hard_sigma = boundary.auxiliary_regions[0]
      .dofs
      .iter()
      .copied()
      .collect::<BTreeSet<_>>();
    let sigma_pred = |kidx: KSimplexIdx| hard_sigma.contains(&kidx);
    let u_pred = |kidx: KSimplexIdx| hard_u.contains(&kidx);
    let sigma_data = |_kidx: KSimplexIdx| 0.0;
    let u_data = |_kidx: KSimplexIdx| 0.0;
    let mut sigma_rhs = Vector::zeros(galmats.sigma_len());
    let mut u_rhs = Vector::zeros(galmats.u_len());
    let reduced = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
      &sigma_pred,
      &sigma_data,
      &u_pred,
      &u_data,
      &mut sigma_rhs,
      &mut u_rhs,
      &Matrix::zeros(slice.layout.reduced_dimension(), 0),
    );
    let (mass_sigma, a12, a21, k_matrix) = split_reduced_mixed_blocks(
      &reduced.0,
      galmats.free_sigma_len(&sigma_pred),
      galmats.free_u_len(&u_pred),
    );
    let sigma_inverse = diag_matrix(&invert_diag(&lumped_diag(&mass_sigma)));
    let expected = add_sparse(&k_matrix, &(&a21 * &sigma_inverse * &scale_matrix(&a12, -1.0)));

    assert_eq!(slice.drift, csr_to_triplet(&expected));
    assert_eq!(slice.layout.reduced_dimension(), galmats.free_u_len(&u_pred));
  }

  #[test]
  fn soft_1form_slice_keeps_full_state_and_emits_constraints() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let soft_edges = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .take(3)
      .map(|simp| simp.kidx)
      .collect::<Vec<_>>();
    let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
      "soft-u",
      soft_edges.clone(),
      vec![1.0, 2.0, 3.0],
      BoundaryTreatment::SoftEssential { variance: 0.25 },
    ));

    let slice = build_spatial_prior_slice_1form(
      &topology,
      &geometry,
      &boundary,
      Hodge1PriorConfig {
        kappa: 1.0,
        tau: 1.0,
        mass_inverse: Hodge1MassInverse::RowSumLumped,
      },
    )
    .expect("soft 1-form slice should assemble");

    assert_eq!(slice.layout.reduced_dimension(), slice.layout.full_dimension);
    assert_eq!(slice.soft_boundary_constraints.len(), 1);
    let constraint = &slice.soft_boundary_constraints[0];
    assert_eq!(constraint.target, vec![1.0, 2.0, 3.0]);
    assert_eq!(constraint.variance, 0.25);
    assert_eq!(constraint.operator.nrows(), soft_edges.len());
    assert_eq!(constraint.operator.ncols(), slice.layout.full_dimension);
  }

  #[test]
  fn mixed_hard_and_soft_1form_slice_reduces_only_hard_dofs() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let boundary_edges = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .map(|simp| simp.kidx)
      .collect::<Vec<_>>();
    let hard = boundary_edges[..1].to_vec();
    let soft = boundary_edges[1..3].to_vec();
    let boundary = BoundarySpec::default()
      .with_state_region(BoundaryRegionSpec::new(
        "hard-u",
        hard.clone(),
        vec![0.0],
        BoundaryTreatment::HardEssential,
      ))
      .with_state_region(BoundaryRegionSpec::new(
        "soft-u",
        soft.clone(),
        vec![4.0, 5.0],
        BoundaryTreatment::SoftEssential { variance: 1e-2 },
      ));

    let slice = build_spatial_prior_slice_1form(
      &topology,
      &geometry,
      &boundary,
      Hodge1PriorConfig {
        kappa: 1.0,
        tau: 1.0,
        mass_inverse: Hodge1MassInverse::RowSumLumped,
      },
    )
    .expect("mixed 1-form slice should assemble");

    assert_eq!(slice.layout.full_dimension - 1, slice.layout.reduced_dimension());
    assert_eq!(slice.soft_boundary_constraints.len(), 1);
    let rows = slice.soft_boundary_constraints[0]
      .operator
      .triplet_iter()
      .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
  }
}
