//! Shared `ort::Session` constructor for [`crate::TextEncoder`].
//!
//! Gated on `feature = "inference"` because every type touched here
//! (`ort::Session`, `ort::ep::*`) only exists when ort is in the
//! dependency graph.

use std::path::Path;

use crate::{
  error::{Error, Result},
  options::Options,
};

/// Build an `ort::Session` from the graph at `path` with the
/// caller-supplied `Options`. Registers any execution providers the
/// caller opted into (`cuda` / `tensorrt` / `directml` / `rocm` /
/// `coreml` Cargo features) before committing the graph file. The
/// implicit CPU EP is always available as the final fallback.
pub(crate) fn build_session(graph: &Path, opts: Options) -> Result<ort::session::Session> {
  use ort::session::Session;

  let level = opts.optimization_level();

  let mut builder = Session::builder()
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source,
    })?
    .with_optimization_level(level)
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?
    .with_intra_threads(opts.threads().intra_threads())
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?
    .with_inter_threads(opts.threads().inter_threads())
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?
    .with_parallel_execution(opts.threads().parallel_execution())
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?;

  let providers = collect_execution_providers();
  if !providers.is_empty() {
    builder = builder
      .with_execution_providers(providers)
      .map_err(|source| Error::LoadGraph {
        path: graph.to_path_buf(),
        source: ort::Error::from(source),
      })?;
  }

  builder
    .commit_from_file(graph)
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source,
    })
}

/// Collect the execution-provider dispatchers active under the
/// current target + feature configuration. Order matters: ort tries
/// each in the supplied list before falling back to the implicit
/// CPU EP, so the first registered EP gets first refusal on each op.
fn collect_execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
  #[allow(unused_mut)]
  let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();

  // TensorRT before CUDA when both are enabled: TensorRT typically
  // beats raw CUDA on supported ops, and the unsupported ones fall
  // back to CUDA's general execution path.
  #[cfg(feature = "tensorrt")]
  {
    providers.push(ort::ep::TensorRT::default().build());
  }
  #[cfg(feature = "cuda")]
  {
    providers.push(ort::ep::CUDA::default().build());
  }
  #[cfg(feature = "directml")]
  {
    providers.push(ort::ep::DirectML::default().build());
  }
  #[cfg(feature = "rocm")]
  {
    providers.push(ort::ep::ROCm::default().build());
  }
  #[cfg(feature = "coreml")]
  {
    providers.push(ort::ep::CoreML::default().build());
  }

  providers
}
