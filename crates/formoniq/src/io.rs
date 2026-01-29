use std::{
  fs::File,
  io::{self, BufWriter, Write},
  path::Path,
};

use ddf::{cochain::Cochain, whitney::form::WhitneyForm};
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexCoords},
  topology::{complex::Complex, handle::SkeletonHandle},
};

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
    let value = whitney.eval_known_cell(cell, &bary).into_grade1();

    let vx = value[0];
    let vy = if value.len() > 1 { value[1] } else { 0.0 };
    let vz = if value.len() > 2 { value[2] } else { 0.0 };
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
