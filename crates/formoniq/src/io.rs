use std::{
  fs::File,
  io::{self, BufWriter, Write},
  path::Path,
};

use ddf::cochain::Cochain;
use manifold::{
  geometry::coord::mesh::MeshCoords,
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
pub fn write_cochain_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  let k = cochain.dim();
  let cell_type = vtk_cell_type(k).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::Other,
      format!("Unsupported cochain degree {k}"),
    )
  })?;

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let skeleton = topology.skeleton(k);
  if cochain.len() != skeleton.len() {
    return Err(io::Error::new(
      io::ErrorKind::Other,
      format!(
        "Cochain length {} does not match skeleton size {} for dim {k}",
        cochain.len(),
        skeleton.len()
      ),
    ));
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

  // Cells
  let nverts_per_cell = k + 1;
  let ncells = skeleton.len();
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  // Data
  writeln!(w, "CELL_DATA {}", ncells)?;
  writeln!(w, "SCALARS {} double 1", data_name)?;
  writeln!(w, "LOOKUP_TABLE default")?;
  for coeff in cochain.coeffs.iter() {
    writeln!(w, "{coeff:.12}")?;
  }

  Ok(())
}

fn write_skeleton_cells(mut w: impl Write, skeleton: SkeletonHandle) -> io::Result<()> {
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
