use proc_macro::TokenStream;

mod progress;

/// Wraps a function with an indicatif progress bar tied to its tracing span.
///
/// `message` and `finished` configure progress-bar text. Any additional
/// arguments are forwarded to `#[instrument(...)]`, and `progress = true` is
/// injected into instrument fields automatically. If no `#[instrument]`
/// attribute is present, one is added automatically.
///
/// `finished` is evaluated when the function body completes. During evaluation,
/// `result` is available as a reference to the function return value.
///
/// Inside the attributed function body, this macro also injects helper
/// macros for incremental progress updates:
///
/// - `progress_set_length!(<usize>)`
/// - `progress_inc_length!(<usize>)`
/// - `progress_inc!()` or `progress_inc!(<usize>)`
/// - `progress_trace!(<completed>, <total>)`
#[proc_macro_attribute]
pub fn progress(attr: TokenStream, item: TokenStream) -> TokenStream {
    progress::expand(attr, item)
}
