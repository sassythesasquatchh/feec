use formoniq::torus_convergence::{
  computed_vector_field_output_path, projected_exact_vector_field_output_path,
  run_torus_convergence,
};

use std::{
  fs,
  path::PathBuf,
  process,
  time::{SystemTime, UNIX_EPOCH},
};

fn unique_output_dir() -> PathBuf {
  let unique = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .expect("system clock before unix epoch")
    .as_nanos();
  std::env::temp_dir().join(format!(
    "formoniq_torus_convergence_outputs_{}_{}",
    process::id(),
    unique
  ))
}

#[test]
fn torus_convergence_writes_vector_field_outputs_for_each_resolution(
) -> Result<(), Box<dyn std::error::Error>> {
  let output_dir = unique_output_dir();

  let results = run_torus_convergence(&output_dir)?;
  assert_eq!(results.len(), 4);

  for resolution in 0..=3 {
    let computed_path = computed_vector_field_output_path(&output_dir, resolution);
    let projected_path = projected_exact_vector_field_output_path(&output_dir, resolution);

    assert!(
      computed_path.exists(),
      "missing computed vector field for resolution {resolution}: {:?}",
      computed_path
    );
    assert!(
      projected_path.exists(),
      "missing projected exact vector field for resolution {resolution}: {:?}",
      projected_path
    );

    let computed_vtk = fs::read_to_string(&computed_path)?;
    let projected_vtk = fs::read_to_string(&projected_path)?;

    assert!(computed_vtk.contains("CELL_DATA"));
    assert!(computed_vtk.contains("VECTORS solution_computed_vector_field double"));
    assert!(projected_vtk.contains("CELL_DATA"));
    assert!(projected_vtk.contains("VECTORS solution_projected_exact_vector_field double"));
  }

  let _ = fs::remove_dir_all(&output_dir);
  Ok(())
}
