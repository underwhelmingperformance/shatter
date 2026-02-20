use tracing::warn;

use super::gpu::ShatterTransitionRenderer;
use super::{RenderReceipt, TransitionError, TransitionRequest};

/// Service interface for transition rendering backends.
pub trait TransitionService: Send + Sync {
    /// Renders one transition request.
    ///
    /// # Errors
    ///
    /// Returns an error when source inputs cannot be decoded, rendering fails,
    /// or output writing fails.
    fn render(&self, request: TransitionRequest) -> Result<RenderReceipt, TransitionError>;
}

/// Production transition service.
///
/// This renderer executes the shatter transition via `wgpu`. It first requests
/// a hardware adapter and then falls back to a `wgpu` fallback adapter when a
/// hardware adapter is unavailable.
#[derive(Debug, Default)]
pub struct RealTransitionService {
    shatter_renderer: ShatterTransitionRenderer,
}

impl RealTransitionService {
    /// Creates a transition service with the default shatter renderer.
    ///
    /// ```
    /// use shatter::RealTransitionService;
    ///
    /// let service = RealTransitionService::new();
    /// let _ = service;
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            shatter_renderer: ShatterTransitionRenderer::new(),
        }
    }
}

impl TransitionService for RealTransitionService {
    fn render(&self, request: TransitionRequest) -> Result<RenderReceipt, TransitionError> {
        let result = self.shatter_renderer.render(&request);
        if let Err(error) = &result {
            warn!(?error, "shatter render failed");
        }
        result
    }
}

pub(super) fn fps_to_delay_centiseconds(fps: u16) -> u16 {
    let rounded = (100.0 / f32::from(fps)).round();
    rounded.max(1.0) as u16
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn fps_to_delay_rounds_to_centiseconds() {
        assert_eq!(6, fps_to_delay_centiseconds(16));
    }
}
