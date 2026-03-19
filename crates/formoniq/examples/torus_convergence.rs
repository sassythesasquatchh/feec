use formoniq::torus_convergence::run_torus_convergence;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  for record in run_torus_convergence("out/examples/torus_convergence")? {
    println!(
      "resolution={} l2={:.3e} rate_l2={:.2} hd={:.3e} rate_hd={:.2}",
      record.resolution, record.l2_error, record.l2_rate, record.hd_error, record.hd_rate,
    );
  }

  Ok(())
}
