use crate::assemble::{
  assemble_whitney_projected_sparse_inverse_galmat,
  assemble_whitney_projected_sparse_inverse_galmat_weighted,
};
use crate::operators::InnerProductWeightClosure;
use crate::problems::hodge_laplace::MixedGalmats;
use crate::problems::laplace_beltrami::LaplaceBeltramiGalmats;
use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Matrix, Vector};
use feg_core::{
  BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, FixedDof, LinearGaussianMeasurementSpec,
  SparseTriplet, SparseTripletMatrix, StateLayout,
};
use manifold::geometry::coord::{mesh::MeshCoords, quadrature::SimplexQuadRule};
use manifold::geometry::metric::mesh::MeshLengths;
use manifold::topology::{complex::Complex, handle::KSimplexIdx};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedLinearPdeSystem {
  pub operator: SparseTripletMatrix,
  pub residual_bias: Vec<f64>,
  pub state_mass: SparseTripletMatrix,
  pub state_mass_inverse: Option<SparseTripletMatrix>,
  pub layout: StateLayout,
  pub forcing_operator: SparseTripletMatrix,
  pub neumann_operator: SparseTripletMatrix,
  pub boundary_measurements: Vec<LinearGaussianMeasurementSpec>,
}

impl ReducedLinearPdeSystem {
  pub fn residual_dimension(&self) -> usize {
    self.operator.nrows()
  }

  pub fn state_dimension(&self) -> usize {
    self.operator.ncols()
  }
}

pub fn build_reduced_laplace_beltrami_system(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &BoundarySpec,
) -> Result<ReducedLinearPdeSystem, String> {
  ensure_no_auxiliary_regions(boundary)?;
  let galmats = LaplaceBeltramiGalmats::compute(topology, geometry);
  let full_operator = galmats.stiffness_csr();
  let full_mass = galmats.mass_csr();
  let layout = build_state_layout(full_operator.nrows(), &boundary.state_regions)?;
  let reduced_operator = reduce_square_with_layout(&full_operator, &layout)?;
  let reduced_mass = reduce_square_with_layout(&full_mass, &layout)?;
  let residual_bias = hard_dirichlet_bias(&full_operator, &layout);
  let residual_dim = reduced_operator.nrows();

  Ok(ReducedLinearPdeSystem {
    operator: csr_to_triplet(&reduced_operator),
    residual_bias,
    state_mass: csr_to_triplet(&reduced_mass),
    state_mass_inverse: None,
    layout: layout.clone(),
    forcing_operator: identity_triplet(residual_dim, -1.0),
    neumann_operator: identity_triplet(residual_dim, -1.0),
    boundary_measurements: soft_boundary_measurements(
      &boundary.state_regions,
      layout.full_dimension,
    )?,
  })
}

pub fn build_reduced_hodge_laplace_1form_system(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &BoundarySpec,
) -> Result<ReducedLinearPdeSystem, String> {
  let galmats = MixedGalmats::compute(topology, geometry, 1);
  let state_mass_inverse = CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
    topology, geometry,
  ));
  build_reduced_hodge_laplace_1form_system_with_galmats(&galmats, boundary, &state_mass_inverse)
}

pub fn build_reduced_weighted_hodge_laplace_1form_system(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  boundary: &BoundarySpec,
) -> Result<ReducedLinearPdeSystem, String> {
  let galmats = MixedGalmats::compute_weighted(topology, geometry, 1, coords, qr.clone(), weight);
  let state_mass_inverse =
    CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
      topology, geometry, coords, qr, weight,
    ));
  build_reduced_hodge_laplace_1form_system_with_galmats(&galmats, boundary, &state_mass_inverse)
}

pub fn build_reduced_hodge_laplace_1form_system_with_galmats(
  galmats: &MixedGalmats,
  boundary: &BoundarySpec,
  state_mass_inverse: &CsrMatrix,
) -> Result<ReducedLinearPdeSystem, String> {
  ensure_no_soft_auxiliary(&boundary.auxiliary_regions)?;
  let context = build_mixed_1form_boundary_context(galmats, boundary)?;
  let (schur, schur_rhs) = schur_reduced_mixed_system(
    galmats,
    &context,
    &Vector::zeros(galmats.sigma_len()),
    &Vector::zeros(galmats.u_len()),
  )?;
  let reduced_mass =
    reduce_square_with_layout(&CsrMatrix::from(galmats.mass_u()), &context.layout)?;
  let reduced_mass_inverse = reduce_square_with_layout(state_mass_inverse, &context.layout)?;

  Ok(ReducedLinearPdeSystem {
    operator: csr_to_triplet(&schur),
    residual_bias: schur_rhs.iter().map(|value| -*value).collect(),
    state_mass: csr_to_triplet(&reduced_mass),
    state_mass_inverse: Some(csr_to_triplet(&reduced_mass_inverse)),
    layout: context.layout.clone(),
    forcing_operator: identity_triplet(context.layout.reduced_dimension(), -1.0),
    neumann_operator: identity_triplet(context.layout.reduced_dimension(), -1.0),
    boundary_measurements: soft_boundary_measurements(
      &boundary.state_regions,
      context.layout.full_dimension,
    )?,
  })
}

pub fn reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
  galmats: &MixedGalmats,
  boundary: &BoundarySpec,
  sigma_rhs: &Vector,
  u_rhs: &Vector,
) -> Result<Vector, String> {
  ensure_no_soft_auxiliary(&boundary.auxiliary_regions)?;
  if sigma_rhs.len() != galmats.sigma_len() {
    return Err(format!(
      "sigma rhs length {} must match sigma dimension {}",
      sigma_rhs.len(),
      galmats.sigma_len()
    ));
  }
  if u_rhs.len() != galmats.u_len() {
    return Err(format!(
      "u rhs length {} must match state dimension {}",
      u_rhs.len(),
      galmats.u_len()
    ));
  }
  let context = build_mixed_1form_boundary_context(galmats, boundary)?;
  let (_, schur_rhs) = schur_reduced_mixed_system(galmats, &context, sigma_rhs, u_rhs)?;
  Ok(schur_rhs)
}

fn ensure_no_auxiliary_regions(boundary: &BoundarySpec) -> Result<(), String> {
  if !boundary.auxiliary_regions.is_empty() {
    return Err("0-form systems do not support auxiliary boundary regions".to_string());
  }
  Ok(())
}

fn ensure_no_soft_auxiliary(regions: &[BoundaryRegionSpec]) -> Result<(), String> {
  if let Some(region) = regions
    .iter()
    .find(|region| matches!(region.treatment, BoundaryTreatment::SoftEssential { .. }))
  {
    return Err(format!(
            "soft auxiliary boundary conditions are not supported for reduced mixed 1-form systems; region '{}' must be hard or natural",
            region.name
        ));
  }
  Ok(())
}

fn build_state_layout(
  full_dimension: usize,
  regions: &[BoundaryRegionSpec],
) -> Result<StateLayout, String> {
  let fixed = collect_fixed_dofs(full_dimension, regions)?;
  layout_from_fixed(full_dimension, &fixed)
}

fn layout_from_fixed(
  full_dimension: usize,
  fixed_dofs: &[FixedDof],
) -> Result<StateLayout, String> {
  let fixed_indices = fixed_dofs
    .iter()
    .map(|entry| entry.index)
    .collect::<BTreeSet<_>>();
  let active_dofs = (0..full_dimension)
    .filter(|index| !fixed_indices.contains(index))
    .collect::<Vec<_>>();
  Ok(StateLayout::new(
    full_dimension,
    active_dofs,
    fixed_dofs.to_vec(),
  ))
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
        return Err(format!(
          "dof {dof} is fixed by multiple hard-essential regions"
        ));
      }
    }
  }
  Ok(
    fixed
      .into_iter()
      .map(|(index, value)| FixedDof { index, value })
      .collect(),
  )
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
        region.name, dof, full_dimension
      ));
    }
  }
  Ok(())
}

fn reduced_index_map(layout: &StateLayout) -> Vec<Option<usize>> {
  let mut map = vec![None; layout.full_dimension];
  for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
    map[full] = Some(reduced);
  }
  map
}

fn reduce_square_with_layout(
  matrix: &CsrMatrix,
  layout: &StateLayout,
) -> Result<CsrMatrix, String> {
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

fn hard_dirichlet_bias(matrix: &CsrMatrix, layout: &StateLayout) -> Vec<f64> {
  let reduced_map = reduced_index_map(layout);
  let mut bias = vec![0.0; layout.reduced_dimension()];
  let fixed_map = layout
    .fixed_dofs
    .iter()
    .map(|entry| (entry.index, entry.value))
    .collect::<BTreeMap<_, _>>();
  for (row, col, value) in matrix.triplet_iter() {
    let Some(reduced_row) = reduced_map[row] else {
      continue;
    };
    if let Some(fixed_value) = fixed_map.get(&col) {
      bias[reduced_row] += *value * *fixed_value;
    }
  }
  bias
}

fn soft_boundary_measurements(
  regions: &[BoundaryRegionSpec],
  full_dimension: usize,
) -> Result<Vec<LinearGaussianMeasurementSpec>, String> {
  let mut measurements = Vec::new();
  for region in regions {
    let BoundaryTreatment::SoftEssential { variance } = region.treatment else {
      continue;
    };
    validate_region(region, full_dimension)?;
    let operator = SparseTripletMatrix::from_triplets(
      region.dofs.len(),
      full_dimension,
      region
        .dofs
        .iter()
        .enumerate()
        .map(|(row, dof)| SparseTriplet {
          row,
          col: *dof,
          value: 1.0,
        }),
    );
    measurements.push(LinearGaussianMeasurementSpec {
      name: region.name.clone(),
      operator,
      observations: region.values.clone(),
      bias: vec![0.0; region.dofs.len()],
      variance,
    });
  }
  Ok(measurements)
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

#[derive(Debug, Clone)]
struct Mixed1FormBoundaryContext {
  layout: StateLayout,
  auxiliary_fixed_map: Vec<Option<f64>>,
  state_fixed_map: Vec<Option<f64>>,
  auxiliary_fixed_set: BTreeSet<usize>,
  state_fixed_set: BTreeSet<usize>,
}

fn build_mixed_1form_boundary_context(
  galmats: &MixedGalmats,
  boundary: &BoundarySpec,
) -> Result<Mixed1FormBoundaryContext, String> {
  let layout = build_state_layout(galmats.u_len(), &boundary.state_regions)?;
  let auxiliary_fixed = collect_fixed_dofs(galmats.sigma_len(), &boundary.auxiliary_regions)?;
  Ok(Mixed1FormBoundaryContext {
    auxiliary_fixed_map: fixed_map(galmats.sigma_len(), &auxiliary_fixed)?,
    state_fixed_map: fixed_map(galmats.u_len(), &layout.fixed_dofs)?,
    auxiliary_fixed_set: auxiliary_fixed.iter().map(|entry| entry.index).collect(),
    state_fixed_set: layout.fixed_dofs.iter().map(|entry| entry.index).collect(),
    layout,
  })
}

fn schur_reduced_mixed_system(
  galmats: &MixedGalmats,
  context: &Mixed1FormBoundaryContext,
  sigma_rhs: &Vector,
  u_rhs: &Vector,
) -> Result<(CsrMatrix, Vector), String> {
  let sigma_predicate = |kidx: KSimplexIdx| context.auxiliary_fixed_set.contains(&kidx);
  let state_predicate = |kidx: KSimplexIdx| context.state_fixed_set.contains(&kidx);
  let sigma_data = |kidx: KSimplexIdx| context.auxiliary_fixed_map[kidx].unwrap_or(0.0);
  let state_data = |kidx: KSimplexIdx| context.state_fixed_map[kidx].unwrap_or(0.0);
  let reduced_sigma_len = galmats.free_sigma_len(&sigma_predicate);
  let reduced_u_len = galmats.free_u_len(&state_predicate);

  let mut sigma_rhs = sigma_rhs.clone();
  let mut u_rhs = u_rhs.clone();
  let (reduced_mixed, reduced_rhs) = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
    &sigma_predicate,
    &sigma_data,
    &state_predicate,
    &state_data,
    &mut sigma_rhs,
    &mut u_rhs,
    &Matrix::zeros(context.layout.reduced_dimension(), 0),
  );
  Ok(schur_reduce_eliminated_mixed_system(
    &reduced_mixed,
    &reduced_rhs,
    reduced_sigma_len,
    reduced_u_len,
  ))
}

fn schur_reduce_eliminated_mixed_system(
  reduced_mixed: &CsrMatrix,
  reduced_rhs: &Vector,
  reduced_sigma_len: usize,
  reduced_u_len: usize,
) -> (CsrMatrix, Vector) {
  let (mass_sigma, a12, a21, k_matrix) =
    split_reduced_mixed_blocks(reduced_mixed, reduced_sigma_len, reduced_u_len);
  let mass_sigma_inv = diag_matrix(&invert_diag(&lumped_diag(&mass_sigma)));
  let schur = if mass_sigma.nrows() == 0 {
    k_matrix
  } else {
    add_sparse(
      &k_matrix,
      &(&a21 * &mass_sigma_inv * &scale_matrix(&a12, -1.0)),
    )
  };

  let (rhs_sigma, rhs_u) = split_reduced_rhs(reduced_rhs, reduced_sigma_len, reduced_u_len);
  let schur_rhs = if mass_sigma.nrows() == 0 {
    rhs_u
  } else {
    let sigma_correction = &mass_sigma_inv * &rhs_sigma;
    rhs_u - &a21 * sigma_correction
  };
  (schur, schur_rhs)
}

fn split_reduced_rhs(
  rhs: &Vector,
  reduced_sigma_len: usize,
  reduced_u_len: usize,
) -> (Vector, Vector) {
  let rhs_sigma = Vector::from_iterator(
    reduced_sigma_len,
    (0..reduced_sigma_len).map(|index| rhs[index]),
  );
  let rhs_u = Vector::from_iterator(
    reduced_u_len,
    (0..reduced_u_len).map(|index| rhs[reduced_sigma_len + index]),
  );
  (rhs_sigma, rhs_u)
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
  let mut coo = CooMatrix::from(lhs);
  for (row, col, value) in rhs.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}

fn identity_triplet(dimension: usize, scale: f64) -> SparseTripletMatrix {
  SparseTripletMatrix::from_triplets(
    dimension,
    dimension,
    (0..dimension).map(|index| SparseTriplet {
      row: index,
      col: index,
      value: scale,
    }),
  )
}

fn csr_to_triplet(matrix: &CsrMatrix) -> SparseTripletMatrix {
  SparseTripletMatrix::from_triplets(
    matrix.nrows(),
    matrix.ncols(),
    matrix
      .triplet_iter()
      .map(|(row, col, value)| SparseTriplet {
        row,
        col,
        value: *value,
      }),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{assemble, operators::InnerProductWeightClosure};
  use manifold::gen::cartesian::CartesianMeshInfo;

  #[test]
  fn reduced_laplace_beltrami_system_emits_soft_boundary_measurements() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let soft = topology
      .boundary_subcomplex_simplices(0)
      .into_iter()
      .take(2)
      .map(|simp| simp.kidx)
      .collect::<Vec<_>>();
    let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
      "soft",
      soft.clone(),
      vec![1.0, 2.0],
      BoundaryTreatment::SoftEssential { variance: 0.5 },
    ));
    let system = build_reduced_laplace_beltrami_system(&topology, &geometry, &boundary)
      .expect("0-form system should assemble");

    assert_eq!(
      system.layout.reduced_dimension(),
      system.layout.full_dimension
    );
    assert_eq!(system.boundary_measurements.len(), 1);
    let measurement = &system.boundary_measurements[0];
    assert_eq!(measurement.observations, vec![1.0, 2.0]);
    assert_eq!(measurement.operator.ncols(), system.layout.full_dimension);
  }

  fn dense_from_triplets(matrix: &SparseTripletMatrix) -> Matrix {
    let mut dense = Matrix::zeros(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
      dense[(row, col)] += value;
    }
    dense
  }

  #[test]
  fn weighted_reduced_hodge_laplace_1form_matches_unweighted_for_unit_weight() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let state_dofs =
      assemble::boundary_simplices_where_barycenter(&topology, &coords, 1, |point| point[0] == 0.0)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    let auxiliary_dofs =
      assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, |point| point[0] == 0.0)
        .into_iter()
        .take(1)
        .collect::<Vec<_>>();
    let boundary = BoundarySpec::default()
      .with_state_region(BoundaryRegionSpec::new(
        "soft-state",
        state_dofs.clone(),
        vec![0.0; state_dofs.len()],
        BoundaryTreatment::SoftEssential { variance: 1e-4 },
      ))
      .with_auxiliary_region(BoundaryRegionSpec::new(
        "hard-aux",
        auxiliary_dofs,
        vec![0.0],
        BoundaryTreatment::HardEssential,
      ));

    let unweighted = build_reduced_hodge_laplace_1form_system(&topology, &geometry, &boundary)
      .expect("unweighted system should assemble");
    let weighted = build_reduced_weighted_hodge_laplace_1form_system(
      &topology,
      &geometry,
      &coords,
      None,
      &InnerProductWeightClosure::new(|_| 1.0),
      &boundary,
    )
    .expect("weighted unit system should assemble");

    assert_eq!(unweighted.layout, weighted.layout);
    assert_eq!(
      unweighted.boundary_measurements,
      weighted.boundary_measurements
    );
    assert_eq!(
      unweighted.residual_bias.len(),
      weighted.residual_bias.len(),
      "residual bias lengths should match"
    );
    assert_eq!(
      unweighted
        .state_mass_inverse
        .as_ref()
        .map(SparseTripletMatrix::nrows),
      weighted
        .state_mass_inverse
        .as_ref()
        .map(SparseTripletMatrix::nrows),
      "state mass inverse dimensions should match"
    );

    let operator_diff =
      dense_from_triplets(&unweighted.operator) - dense_from_triplets(&weighted.operator);
    let mass_diff =
      dense_from_triplets(&unweighted.state_mass) - dense_from_triplets(&weighted.state_mass);
    let mass_inverse_diff = dense_from_triplets(
      unweighted
        .state_mass_inverse
        .as_ref()
        .expect("unweighted 1-form system should expose a reduced mass inverse"),
    ) - dense_from_triplets(
      weighted
        .state_mass_inverse
        .as_ref()
        .expect("weighted unit 1-form system should expose a reduced mass inverse"),
    );
    let bias_diff = Vector::from_vec(
      unweighted
        .residual_bias
        .iter()
        .zip(weighted.residual_bias.iter())
        .map(|(left, right)| left - right)
        .collect(),
    );

    assert!(operator_diff.norm() <= 1e-10);
    assert!(mass_diff.norm() <= 1e-10);
    assert!(mass_inverse_diff.norm() <= 1e-10);
    assert!(bias_diff.norm() <= 1e-10);
  }

  #[test]
  fn weighted_reduced_hodge_laplace_1form_exposes_reduced_nc1_inverse() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let boundary = BoundarySpec::default();
    let weight = InnerProductWeightClosure::new(|point| 1.0 + point[0] + 0.5 * point[1]);

    let system = build_reduced_weighted_hodge_laplace_1form_system(
      &topology, &geometry, &coords, None, &weight, &boundary,
    )
    .expect("weighted 1-form system should assemble");

    let reduced_inverse = system
      .state_mass_inverse
      .as_ref()
      .expect("weighted 1-form system should expose a reduced NC1 projected inverse");
    assert_eq!(reduced_inverse.nrows(), system.state_dimension());
    assert_eq!(reduced_inverse.ncols(), system.state_dimension());
    assert!(reduced_inverse
      .triplet_iter()
      .all(|(_, _, value)| value.is_finite()));
  }
}
