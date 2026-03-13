use std::{
  fs::{self, File},
  io::{self, BufWriter, Write},
  path::Path,
};

use common::linalg::nalgebra::Vector;
use ddf::{
  cochain::Cochain,
  whitney::{form::WhitneyForm, lsf::WhitneyLsf},
};
use exterior::field::ExteriorField;
use manifold::{
  geometry::coord::{
    mesh::MeshCoords,
    simplex::{barycenter_local, SimplexCoords},
  },
  topology::{
    complex::Complex,
    handle::{SimplexHandle, SkeletonHandle},
  },
};

pub fn write_cochain(path: &str, cochain: &Cochain) -> std::io::Result<()> {
  let mut file = fs::File::create(path)?;
  for coeff in cochain.coeffs.iter() {
    writeln!(file, "{coeff:.12}")?;
  }
  Ok(())
}

fn vtk_cell_type(k: usize) -> Option<u32> {
  match k {
    0 => Some(1),  // VTK_VERTEX
    1 => Some(3),  // VTK_LINE
    2 => Some(5),  // VTK_TRIANGLE
    3 => Some(10), // VTK_TETRA
    _ => None,
  }
}

/// Write a cochain to a legacy VTK (ASCII) unstructured grid.
///
/// The cochain degree determines which simplices are written:
/// - 0-cochains -> vertices as 1-vertex cells
/// - 1-cochains -> edges as line cells
/// - 2-cochains -> triangles
/// - 3-cochains -> tetrahedra
///
/// Coordinates with dimension < 3 are zero-padded to 3D for VTK.
// pub fn write_cochain_vtk(
//   path: impl AsRef<Path>,
//   coords: &MeshCoords,
//   topology: &Complex,
//   cochain: &Cochain,
//   data_name: &str,
// ) -> io::Result<()> {
//   let k = cochain.dim();
//   let cell_type = vtk_cell_type(k).ok_or_else(|| {
//     io::Error::new(
//       io::ErrorKind::Other,
//       format!("Unsupported cochain degree {k}"),
//     )
//   })?;

//   if coords.dim() > 3 {
//     return Err(io::Error::new(
//       io::ErrorKind::Other,
//       "VTK export supports up to 3D coordinates",
//     ));
//   }

//   let skeleton = topology.skeleton(k);
//   if cochain.len() != skeleton.len() {
//     return Err(io::Error::new(
//       io::ErrorKind::Other,
//       format!(
//         "Cochain length {} does not match skeleton size {} for dim {k}",
//         cochain.len(),
//         skeleton.len()
//       ),
//     ));
//   }

//   let file = File::create(path)?;
//   let mut w = BufWriter::new(file);

//   writeln!(w, "# vtk DataFile Version 4.2")?;
//   writeln!(w, "{data_name}")?;
//   writeln!(w, "ASCII")?;
//   writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

//   // Points
//   writeln!(w, "POINTS {} double", coords.nvertices())?;
//   for coord in coords.coord_iter() {
//     let x = coord[0];
//     let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
//     let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
//     writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
//   }

//   // Cells
//   let nverts_per_cell = k + 1;
//   let ncells = skeleton.len();
//   writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
//   write_skeleton_cells(&mut w, skeleton)?;

//   writeln!(w, "CELL_TYPES {}", ncells)?;
//   for _ in 0..ncells {
//     writeln!(w, "{cell_type}")?;
//   }

//   // Data
//   writeln!(w, "CELL_DATA {}", ncells)?;
//   writeln!(w, "SCALARS {} double 1", data_name)?;
//   writeln!(w, "LOOKUP_TABLE default")?;
//   for coeff in cochain.coeffs.iter() {
//     writeln!(w, "{coeff:.12}")?;
//   }

//   Ok(())
// }

pub fn write_cochain_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  let k = cochain.dim();

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      "VTK export supports up to 3D coordinates",
    ));
  }

  // Choose what cells define the *rendered geometry*.
  //
  // - For 0-cochains: export the top-dimensional cells (triangles/tets/...) so ParaView
  //   can shade a continuous surface/volume using POINT_DATA.
  // - For k>0: export the k-skeleton so the dofs live on edges/faces/etc as CELL_DATA.

  let topo_dim = topology.dim();
  let geom_k = if k == 0 { topo_dim } else { k };

  let cell_type = vtk_cell_type(geom_k).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::Other,
      format!("Unsupported cell dimension {geom_k}"),
    )
  })?;

  // Geometry cells we will write
  let geom_skeleton = topology.skeleton(geom_k);

  // Validate cochain length against where the dofs live
  if k == 0 {
    // 0-cochain should match the number of points (vertices)
    if cochain.len() != coords.nvertices() {
      return Err(io::Error::new(
        io::ErrorKind::Other,
        format!(
          "0-cochain length {} does not match number of vertices {}",
          cochain.len(),
          coords.nvertices()
        ),
      ));
    }
  } else {
    // k>0 cochain matches the number of k-cells
    let k_skeleton = topology.skeleton(k);
    if cochain.len() != k_skeleton.len() {
      return Err(io::Error::new(
        io::ErrorKind::Other,
        format!(
          "Cochain length {} does not match skeleton size {} for dim {k}",
          cochain.len(),
          k_skeleton.len()
        ),
      ));
    }
  }

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{data_name}")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  // Points
  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }

  // Cells (geometry)
  let nverts_per_cell = geom_k + 1;
  let ncells = geom_skeleton.len();
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &geom_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  // Data
  if k == 0 {
    // Per-vertex scalars -> continuous shading on polygonal geometry
    writeln!(w, "POINT_DATA {}", coords.nvertices())?;
    writeln!(w, "SCALARS {} double 1", data_name)?;
    writeln!(w, "LOOKUP_TABLE default")?;
    for coeff in cochain.coeffs.iter() {
      writeln!(w, "{coeff:.12}")?;
    }
  } else {
    // k-cell scalars (edges for k=1, faces for k=2, ...)
    writeln!(w, "CELL_DATA {}", ncells)?;
    writeln!(w, "SCALARS {} double 1", data_name)?;
    writeln!(w, "LOOKUP_TABLE default")?;
    for coeff in cochain.coeffs.iter() {
      writeln!(w, "{coeff:.12}")?;
    }
  }

  Ok(())
}

/// Sample a Whitney 1-form at cell barycenters and export as a vector field to VTK (CELL_DATA).
///
/// - The provided cochain must have degree 1.
/// - The vectors are piecewise constant: one vector per top-dimensional cell, evaluated at its barycenter.
/// - For embedded meshes (`topology.dim() < coords.dim()`), we evaluate the Whitney form in local
///   cell coordinates and lift the result to ambient coordinates via the cell pseudoinverse transpose.
pub fn write_1form_vector_field_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  if cochain.dim() != 1 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!("Expected a 1-cochain, got dim {}", cochain.dim()),
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let topo_dim = topology.dim();
  if topo_dim > coords.dim() {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!(
        "Invalid mesh dimensions: topology dim {} > coordinate dim {}",
        topo_dim,
        coords.dim()
      ),
    ));
  }

  let edge_skeleton = topology.skeleton(1);
  if cochain.len() != edge_skeleton.len() {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!(
        "Cochain length {} does not match edge skeleton size {}",
        cochain.len(),
        edge_skeleton.len()
      ),
    ));
  }

  let cell_type = vtk_cell_type(topo_dim).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::Other,
      format!("Unsupported cell dimension {topo_dim}"),
    )
  })?;

  let geom_skeleton = topology.skeleton(topo_dim);

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{data_name}")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  // Points
  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }

  // Cells
  let nverts_per_cell = topo_dim + 1;
  let ncells = geom_skeleton.len();
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &geom_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  // Data: piecewise-constant vectors per top cell
  let whitney =
    (topo_dim == coords.dim()).then(|| WhitneyForm::new(cochain.clone(), topology, coords));

  writeln!(w, "CELL_DATA {}", ncells)?;
  writeln!(w, "VECTORS {} double", data_name)?;

  for cell in geom_skeleton.handle_iter() {
    let value = if let Some(whitney) = &whitney {
      let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
      let bary = cell_coords.barycenter();
      whitney.eval_known_cell(cell, &bary).into_grade1()
    } else {
      eval_embedded_1form_cell_vector(cell, coords, cochain)?
    };

    let vx = value[0];
    let vy = if value.len() > 1 { value[1] } else { 0.0 };
    let vz = if value.len() > 2 { value[2] } else { 0.0 };
    writeln!(w, "{vx:.12} {vy:.12} {vz:.12}")?;
  }

  Ok(())
}

fn eval_embedded_1form_cell_vector(
  cell: SimplexHandle<'_>,
  coords: &MeshCoords,
  cochain: &Cochain,
) -> io::Result<Vector> {
  let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
  let intrinsic_dim = cell_coords.dim_intrinsic();
  let ambient_dim = cell_coords.dim_ambient();
  if intrinsic_dim >= ambient_dim {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!(
        "Expected embedded cell with intrinsic dim < ambient dim, got {} and {}",
        intrinsic_dim, ambient_dim
      ),
    ));
  }

  let bary_local = barycenter_local(intrinsic_dim);
  let mut local_value = Vector::zeros(intrinsic_dim);
  for dof_simp in cell.mesh_subsimps(1) {
    let local_dof_simp = dof_simp.relative_to(&cell);
    let lsf = WhitneyLsf::standard(intrinsic_dim, local_dof_simp);
    let lsf_value = lsf.at_point(&bary_local).into_grade1();
    local_value += cochain[dof_simp] * lsf_value;
  }

  let jacobian_pinv = cell_coords.inv_linear_transform();
  Ok(jacobian_pinv.transpose() * local_value)
}

/// Write vector proxies for a 1-form as edge-aligned vectors (CELL_DATA on the 1-skeleton).
///
/// Each proxy is computed as:
///   v = (cochain(edge) / |edge|) * edge_direction
/// which yields a piecewise-constant vector along each oriented edge.
pub fn write_1form_vector_proxy_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  if cochain.dim() != 1 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!("Expected a 1-cochain, got dim {}", cochain.dim()),
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let edge_skeleton = topology.skeleton(1);
  if cochain.len() != edge_skeleton.len() {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!(
        "Cochain length {} does not match edge skeleton size {}",
        cochain.len(),
        edge_skeleton.len()
      ),
    ));
  }

  let cell_type = vtk_cell_type(1)
    .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Unsupported edge cell type"))?;

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{data_name}")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  // Points
  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }

  // Cells: edges
  let nverts_per_cell = 2;
  let ncells = edge_skeleton.len();
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &edge_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  // Data: edge-aligned vector proxies
  writeln!(w, "CELL_DATA {}", ncells)?;
  writeln!(w, "VECTORS {} double", data_name)?;

  for edge in edge_skeleton.handle_iter() {
    let v0 = coords.coord(edge.vertices[0]);
    let v1 = coords.coord(edge.vertices[1]);
    let mut dir = (v1 - v0).into_owned();
    let length = dir.norm();
    if length > 0.0 {
      let scale = cochain[edge] / length;
      dir *= scale;
    } else {
      dir.fill(0.0);
    }

    let vx = dir[0];
    let vy = if dir.len() > 1 { dir[1] } else { 0.0 };
    let vz = if dir.len() > 2 { dir[2] } else { 0.0 };
    writeln!(w, "{vx:.12} {vy:.12} {vz:.12}")?;
  }

  Ok(())
}

/// Sample a Whitney 2-form at cell barycenters, Hodge-dual it to a vector field,
/// and export as VTK (CELL_DATA).
///
/// Assumptions:
/// - The provided cochain has degree 2 in a 3D mesh.
/// - Coordinates are Euclidean; the Hodge dual reduces to the standard
///   pseudovector mapping: (c01, c02, c12) -> (c12, -c02, c01).
pub fn write_2form_vector_field_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  if cochain.dim() != 2 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!("Expected a 2-cochain, got dim {}", cochain.dim()),
    ));
  }

  if coords.dim() != 3 || topology.dim() != 3 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      "write_2form_vector_field_vtk supports 3D meshes only",
    ));
  }

  let topo_dim = topology.dim();
  let cell_type = vtk_cell_type(topo_dim).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::Other,
      format!("Unsupported cell dimension {topo_dim}"),
    )
  })?;

  let geom_skeleton = topology.skeleton(topo_dim);

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{data_name}")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  // Points
  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }

  // Cells
  let nverts_per_cell = topo_dim + 1;
  let ncells = geom_skeleton.len();
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &geom_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  // Data: piecewise-constant vectors per top cell
  let whitney = WhitneyForm::new(cochain.clone(), topology, coords);

  writeln!(w, "CELL_DATA {}", ncells)?;
  writeln!(w, "VECTORS {} double", data_name)?;

  for cell in geom_skeleton.handle_iter() {
    let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
    let bary = cell_coords.barycenter();
    let value = whitney.eval_known_cell(cell, &bary);

    // value is a grade-2 element in 3D: coeffs correspond to (0,1), (0,2), (1,2)
    let coeffs = value.coeffs();
    assert!(
      coeffs.len() == 3,
      "Expected 3 coefficients for a 2-form in 3D"
    );
    let c01 = coeffs[0];
    let c02 = coeffs[1];
    let c12 = coeffs[2];

    let vx = c12; // dy^dz term -> x component
    let vy = -c02; // dx^dz term -> -y component (orientation)
    let vz = c01; // dx^dy term -> z component

    writeln!(w, "{vx:.12} {vy:.12} {vz:.12}")?;
  }

  Ok(())
}

fn write_skeleton_cells(mut w: impl Write, skeleton: &SkeletonHandle) -> io::Result<()> {
  let nverts_per_cell = skeleton.dim() + 1;
  for simplex in skeleton.handle_iter() {
    write!(w, "{nverts_per_cell}")?;
    for &vertex in simplex.vertices.iter() {
      write!(w, " {}", vertex)?;
    }
    writeln!(w)?;
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use common::linalg::nalgebra::Vector;
  use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::mesh::standard_coord_complex,
  };

  #[test]
  fn write_1form_vector_proxy_vtk_smoke() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let path = std::env::temp_dir().join("proxy_1form.vtk");
    write_1form_vector_proxy_vtk(&path, &coords, &topology, &cochain, "proxy").unwrap();
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_1form_vector_field_vtk_embedded_surface_smoke() {
    let (topology, coords_2d) = standard_coord_complex(2);
    let coords = coords_2d.embed_euclidean(3);
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let path = std::env::temp_dir().join("embedded_1form_vector_field.vtk");
    write_1form_vector_field_vtk(&path, &coords, &topology, &cochain, "embedded").unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let vectors_idx = lines
      .iter()
      .position(|line| line.trim() == "VECTORS embedded double")
      .unwrap();
    let vector_line = lines[vectors_idx + 1];
    let comps: Vec<f64> = vector_line
      .split_whitespace()
      .map(|value| value.parse::<f64>().unwrap())
      .collect();
    assert_eq!(comps.len(), 3);
    assert!(comps.iter().all(|value| value.is_finite()));
    assert!(comps[2].abs() < 1e-10);

    std::fs::remove_file(path).ok();
  }
}
