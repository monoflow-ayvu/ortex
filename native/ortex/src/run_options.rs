//! Per-run controls, shared with the BEAM so a run can be cancelled from another process.

use ort::session::RunOptions;
use rustler::{Resource, ResourceArc};

/// A `RunOptions` handle the BEAM owns.
///
/// `ort`'s `RunOptions` is `Send + Sync` (with no output selection), and `terminate()` takes
/// `&self`: the intent is exactly this - one thread runs the session while another flips the
/// termination flag, and ONNX Runtime stops the graph at the next opportunity. That is the
/// only way out of a run wedged in a driver or an NPU in a bad state, which otherwise holds
/// its session (and, with `sessions: 1`, every later inference) forever.
///
/// One handle per `Ortex.RunOptions.new/1`; passing it to several runs means terminating it
/// terminates all of them.
pub struct RunOptionsResource(pub RunOptions);

#[rustler::resource_impl(name = "OrtexRunOptions")]
impl Resource for RunOptionsResource {}

// `RunOptions` holds a raw ORT pointer, so it is not unwind-safe; encoding only touches the
// resource handle. Same reasoning as `OrtexModel` and `TensorResource`.
impl std::panic::RefUnwindSafe for RunOptionsResource {}
impl std::panic::UnwindSafe for RunOptionsResource {}

impl RunOptionsResource {
    pub fn new(tag: Option<String>) -> rustler::NifResult<ResourceArc<Self>> {
        let options = match tag {
            Some(tag) => RunOptions::new()
                .and_then(|options| options.with_tag(tag))
                .map_err(crate::error::OrtexError::onnxruntime)?,
            None => RunOptions::new().map_err(crate::error::OrtexError::onnxruntime)?,
        };

        Ok(ResourceArc::new(RunOptionsResource(options)))
    }
}
