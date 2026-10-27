//! One error type for everything the NIF returns.
//!
//! Rustler's `Error::Term` becomes `{:error, term}` on the Elixir side, so the term is the
//! API: `{reason, provider, detail}`. `reason` is a closed set of atoms, `provider` names
//! the execution provider when the failure is about one (`nil` otherwise) and `detail` is
//! the human-readable message ORT or this crate produced. `Ortex.Error` is the Elixir face
//! of that tuple; anything opaque (a bare string) would force callers back to matching on
//! messages.

use core::fmt::Display;
use rustler::Atom;

use crate::constants::ortex_atoms;

/// A NIF failure, tagged so the BEAM can act on it instead of reading a string.
#[derive(Debug)]
pub struct OrtexError {
    /// One of `:onnxruntime`, `:invalid_option`, `:invalid_input`,
    /// `:execution_provider`, `:qnn`.
    pub reason: Atom,
    /// The execution provider the failure is about, when there is one.
    pub provider: Option<Atom>,
    /// What went wrong, in terms a caller can fix.
    pub detail: String,
}

impl OrtexError {
    /// onnxruntime itself refused the call (session build, run, type conversion).
    pub fn onnxruntime(detail: impl Display) -> Self {
        Self {
            reason: ortex_atoms::onnxruntime(),
            provider: None,
            detail: detail.to_string(),
        }
    }

    /// A load option was present but unusable (`sessions: "abc"`).
    pub fn invalid_option(detail: impl Display) -> Self {
        Self {
            reason: ortex_atoms::invalid_option(),
            provider: None,
            detail: detail.to_string(),
        }
    }

    /// The BEAM handed over something the model cannot accept: input count, slice rank,
    /// a dtype with no conversion.
    pub fn invalid_input(detail: impl Display) -> Self {
        Self {
            reason: ortex_atoms::invalid_input(),
            provider: None,
            detail: detail.to_string(),
        }
    }

    /// The requested execution provider is not in this build - the one case where
    /// `provider` is always known and the fix (rebuild with a cargo feature) is specific.
    pub fn execution_provider(provider: Atom, detail: impl Display) -> Self {
        Self {
            reason: ortex_atoms::execution_provider(),
            provider: Some(provider),
            detail: detail.to_string(),
        }
    }

    /// The QNN *plugin* library, its backend or the device it would select.
    pub fn qnn(detail: impl Display) -> Self {
        Self {
            reason: ortex_atoms::qnn(),
            provider: Some(ortex_atoms::qnn()),
            detail: detail.to_string(),
        }
    }
}

/// `ort` errors are generic over the value a failed call would have returned
/// (`Error<SessionBuilder>` and friends), so the conversion has to be too.
impl<R> From<ort::Error<R>> for OrtexError {
    fn from(e: ort::Error<R>) -> Self {
        OrtexError::onnxruntime(e)
    }
}

impl From<OrtexError> for rustler::Error {
    fn from(e: OrtexError) -> Self {
        rustler::Error::Term(Box::new((e.reason, e.provider, e.detail)))
    }
}

impl Display for OrtexError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.detail)
    }
}
