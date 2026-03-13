use formoniq::{
  io::{write_1form_vector_field_vtk, write_2form_vector_field_vtk, write_cochain_vtk},
  operators::InnerProductWeightClosure,
};
use manifold::topology::handle::KSimplexIdx;

use {
  common::linalg::nalgebra::Vector,
  exterior::field::DiffFormClosure,
  formoniq::{assemble::assemble_galvec, operators::SourceElVec, problems::hodge_laplace},
  manifold::geometry::coord::CoordRef,
};

use std::{collections::HashSet, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let path = "out/examples/torus_convergence";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let minor_radius = 0.3;
  let major_radius = 1.0;

  //-------------------- Exact Solution -----------------------------

  let u_0 = |theta:f64, phi:f64| {
    (2.*theta).sin() * (3.*phi).cos()
  }

  let v_0 = |theta:f64, phi:f64|{
    theta.cos()*(2.*phi).sin()
  }

  let rho = move| theta:f64|{
    major_radius+minor_radius*theta.cos()
  }

  let u_exact = DiffFormClosure::one_form(
    move|p|{

        let x = p[0];
        let y = p[1];
        let z= p[2];

        let s = (x*x + y*y).sqrt();
        



        _rho = rho(theta)

    },
    dim= 3// should it be 2?
  )

  Ok(())
}
